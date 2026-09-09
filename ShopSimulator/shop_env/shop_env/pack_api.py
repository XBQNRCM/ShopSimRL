import atexit
import sys
import logging
import os
from pathlib import Path
from typing import Any, List

from flask import Flask, request, jsonify, Response, send_from_directory

SHOP_ENV_ROOT = Path(__file__).resolve().parents[1]
if str(SHOP_ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(SHOP_ENV_ROOT))

try:
    from .replay import (
        ReplayError,
        list_runs,
        load_episode,
        load_run,
        resolve_run_path,
    )
    from .shop_agent import shop_agent
    from .slot_lease_pool import LeaseMismatchError, SlotLeasePool
except ImportError:  # Support direct ``python shop_env/pack_api.py`` execution.
    from replay import ReplayError, list_runs, load_episode, load_run, resolve_run_path
    from shop_agent import shop_agent
    from slot_lease_pool import LeaseMismatchError, SlotLeasePool
from web_agent_site.utils import DEBUG_PROD_SIZE
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
from web_agent_site.engine.config import ENVIRONMENT_VERSION, PRIMARY_REWARD
from web_agent_site.engine.observation import OBSERVATION_VERSION

# Constants
LOG_FILE = "shop_agent.log"
DEFAULT_ENV_MAX_NUM = int(os.environ.get("SHOPSIM_ENV_SLOTS", "20"))
SERVER_HOST = '0.0.0.0'
SERVER_PORT = int(os.environ.get("SHOPSIM_PORT", "5700"))
DEBUG_UI_DIR = Path(__file__).resolve().parent / "debug_ui"
PROJECT_ROOT = SHOP_ENV_ROOT.parents[1]
_runs_root = Path(os.environ.get("SHOPSIM_RUNS_ROOT", PROJECT_ROOT / "runs"))
RUNS_ROOT = (
    _runs_root.resolve()
    if _runs_root.is_absolute()
    else (PROJECT_ROOT / _runs_root).resolve()
)

# Global variables
envs: List[Any] = []
env_max_num: int = DEFAULT_ENV_MAX_NUM
slot_pool = SlotLeasePool(env_max_num)

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route('/debug-ui', strict_slashes=False, methods=['GET'])
def debug_ui() -> Response:
    """Serve the local Agent-view environment inspector."""
    return send_from_directory(DEBUG_UI_DIR, 'index.html', max_age=0)


@app.route('/debug-ui/<path:filename>', methods=['GET'])
def debug_ui_asset(filename: str) -> Response:
    """Serve version-controlled assets for the local inspector."""
    return send_from_directory(DEBUG_UI_DIR, filename, max_age=0)


@app.route('/replay-ui', strict_slashes=False, methods=['GET'])
def replay_ui() -> Response:
    """Serve the local ShopSimRL experiment replay dashboard."""
    return send_from_directory(DEBUG_UI_DIR, 'replay.html', max_age=0)


def _replay_run_path() -> Path:
    return resolve_run_path(
        request.args.get('path', ''),
        project_root=PROJECT_ROOT,
        runs_root=RUNS_ROOT,
    )


@app.route('/api/replay/runs', methods=['GET'])
def api_replay_runs() -> Response:
    return jsonify({
        'runs_root': str(RUNS_ROOT),
        'runs': list_runs(project_root=PROJECT_ROOT, runs_root=RUNS_ROOT),
    })


@app.route('/api/replay/run', methods=['GET'])
def api_replay_run() -> Response:
    try:
        path = _replay_run_path()
        return jsonify(load_run(path, project_root=PROJECT_ROOT))
    except ReplayError as exc:
        return jsonify({'error': str(exc)}), 400


@app.route('/api/replay/episode', methods=['GET'])
def api_replay_episode() -> Response:
    try:
        path = _replay_run_path()
        episode_id = request.args.get('episode_id', '').strip()
        if not episode_id:
            raise ReplayError('episode_id cannot be empty')
        product_catalog = envs[0].server.product_item_dict if envs else None
        return jsonify(
            load_episode(path, episode_id, product_catalog=product_catalog)
        )
    except ReplayError as exc:
        return jsonify({'error': str(exc)}), 400


@app.route('/api/shop_agent', methods=['POST'])
def api_shop_agent() -> Response:
    """
    API endpoint for shop agent operations.

    Handles three types of actions:
    - release_all: Release all environments
    - release_one: Release a specific environment
    - reset/interact: Process shop agent actions

    Returns:
        JSON response with result or error message
    """
    data = request.get_json(silent=True)
    if data is None:
        logger.error("[Error] No JSON data provided in request")
        return jsonify({'result': {"error": "No JSON data provided"}})

    action = data.get('action')
    env_idx = data.get('env_idx', None)
    response = data.get('response', None)
    idx = data.get('idx', None)
    lease_id = data.get('lease_id')
    if_persona = data.get('if_persona', False)
    include_private_goal = data.get('include_private_goal', False)
    interaction_mode = data.get('interaction_mode', 'single_turn')
    termination_reason = data.get('termination_reason', 'action_limit')
    acquired_here = False
    try:
        # Release all environments
        if action == 'release_all':
            for env in envs:
                env.clear_session()
            slot_pool.reset(env_max_num)
            logger.info("[Init] All environments have been initialized")
            return jsonify({'result': {"message": "All environments have been initialized"}})

        # Release one environment
        if action == 'release_one':
            if (
                env_idx is not None
                and isinstance(env_idx, int)
                and not isinstance(env_idx, bool)
                and lease_id
            ):
                if not slot_pool.validate(env_idx, lease_id):
                    raise LeaseMismatchError(
                        f"stale or invalid lease for environment {env_idx}"
                    )
                envs[env_idx].clear_session()
                was_leased = slot_pool.release(env_idx, lease_id)
                if was_leased:
                    logger.info(f"[Release] Environment {env_idx} has been released")
                    return jsonify({'result': {"message": f"Environment {env_idx} has been released"}})
                else:
                    logger.warning(f"[Release] Environment {env_idx} is already free, no need to release again")
                    return jsonify({'result': {"message": f"Environment {env_idx} is already free"}})
            else:
                logger.error("[Error] release_one requires env_idx and lease_id")
                return jsonify({'result': {
                    "error": "release_one requires a valid env_idx and lease_id"
                }}), 400

        if action == 'reset':
            if env_idx is not None:
                return jsonify({'result': {
                    'error': 'reset must not include env_idx; the server assigns a slot'
                }}), 400
            env_idx = slot_pool.acquire()
            if env_idx is None:
                logger.info("[Wait] No environment slot is available; queuing reset")
                env_idx = slot_pool.acquire(block=True)
            acquired_here = True
            lease_id = slot_pool.lease_id(env_idx)

        elif action in {'interact', 'terminate'}:
            if (
                isinstance(env_idx, bool)
                or not isinstance(env_idx, int)
                or not lease_id
            ):
                return jsonify({'result': {
                    'error': f'{action} requires a valid env_idx and lease_id'
                }}), 400
            if not slot_pool.validate(env_idx, lease_id):
                return jsonify({'result': {
                    'error': f'stale or invalid lease for environment {env_idx}'
                }}), 409
        else:
            return jsonify({'result': {
                'error': f'unsupported action: {action!r}'
            }}), 400

        # Call shop_agent function
        result = shop_agent(
            envs[env_idx],
            env_idx,
            action,
            idx,
            response,
            if_persona=bool(if_persona),
            include_private_goal=bool(include_private_goal),
            interaction_mode=str(interaction_mode),
            termination_reason=str(termination_reason),
        )
        result['lease_id'] = lease_id

        # The caller owns the lease until release_one.  Auto-releasing here
        # races with the caller's finally-release: another worker can lease
        # this slot between the two releases and then have its active lease
        # accidentally freed by the previous worker.
        if 'over' in result and result['over']:
            logger.info(
                f"[Task Over] Environment {env_idx} is awaiting explicit release"
            )

    except LeaseMismatchError as e:
        logger.warning(f"[Lease] {e}")
        return jsonify({'result': {'error': str(e)}}), 409
    except Exception as e:
        if acquired_here and env_idx is not None:
            try:
                envs[env_idx].clear_session()
                slot_pool.release(env_idx, lease_id)
            except (LeaseMismatchError, ValueError):
                logger.exception("Failed to roll back a newly acquired slot")
        logger.exception(f"[Exception] Exception occurred while processing request: {str(e)}")
        return jsonify({'result': {'error': str(e)}}), 500

    return jsonify({'result': result})


@app.route('/healthz', methods=['GET'])
def healthz() -> Response:
    return jsonify({
        'ready': len(envs) == env_max_num,
        'environment_slots': env_max_num,
        'free_slots': len(slot_pool.free_slots()),
        'tasks': len(envs[0].server.goals) if envs else 0,
        'environment_version': (
            envs[0].server.environment_version if envs else ENVIRONMENT_VERSION
        ),
        'primary_reward': PRIMARY_REWARD,
        'observation_version': OBSERVATION_VERSION,
        'debug_ui': '/debug-ui',
        'replay_ui': '/replay-ui',
    })


def initialize_environments() -> None:
    """
    Initialize all environments and add them to the free environment index.
    """
    global envs, env_max_num

    close_environments()
    envs = []
    slot_pool.reset(env_max_num)

    shared_server = None
    for i in range(env_max_num):
        logger.info(f"Environment {i} is being initialized")
        env = WebAgentTextEnv(
            observation_mode='structured_text',
            split="train",
            num_products=DEBUG_PROD_SIZE,
            server=shared_server,
            session_prefix=f"slot-{i}",
        )
        if shared_server is None:
            shared_server = env.server
        envs.append(env)


def close_environments() -> None:
    """Close each shared server once and discard all slot state."""
    global envs
    servers = {id(env.server): env.server for env in envs}
    for server in servers.values():
        server.close()
    envs = []


atexit.register(close_environments)


if __name__ == '__main__':
    initialize_environments()
    app.run(host=SERVER_HOST, port=SERVER_PORT, threaded=True)
