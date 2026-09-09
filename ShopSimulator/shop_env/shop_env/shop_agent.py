import logging
import re
from typing import Dict, Any, Optional

from web_agent_site.engine.config import ENVIRONMENT_VERSION
from web_agent_site.engine.observation import render_observation_state

LOG_FILE = "shop_agent.log"

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


def _canonical_observation(env: Any) -> tuple[Dict[str, Any], str]:
    """Build state once and derive the only public page text from that state."""
    state = env.structured_observation()
    return state, render_observation_state(state)


def _handle_reset_action(
    env: Any,
    env_idx: int,
    task_idx: Optional[int],
    if_persona: bool = False,
    include_private_goal: bool = False,
    interaction_mode: str = "single_turn",
) -> Dict[str, Any]:
    """
    Handle reset action

    Args:
        env: Environment object
        env_idx: Environment index
        task_idx: Task index

    Returns:
        Dictionary containing reset information
    """
    logger.info(f"[Reset] Starting task {task_idx}, environment index: {env_idx}")
    message = f"Task {task_idx} started"
    env.reset(
        idx=task_idx,
        if_persona=if_persona,
        interaction_mode=interaction_mode,
    )
    observation_state, observation = _canonical_observation(env)
    return_info = {
        'task_instruction': env.instruction_text,
        'observation': observation,
        'message': message,
        'env_idx': env_idx,
        'idx': task_idx,
        'task_mode': 'persona' if env.if_persona else 'standard',
        'interaction_mode': interaction_mode,
        'environment_version': getattr(
            env.server,
            "environment_version",
            ENVIRONMENT_VERSION,
        ),
        "observation_state": observation_state,
    }
    if include_private_goal:
        goal_lookup = getattr(env.server, "goals_by_task_id", None)
        private_goal = (
            goal_lookup[task_idx]
            if goal_lookup is not None
            else env.server.goals[task_idx]
        )
        return_info['private_task_instruction'] = private_goal['instruction_text']
        return_info['goal_options'] = private_goal['goal_options']
        return_info['reason_key'] = private_goal.get('reason_key')
    if hasattr(env, 'user_persona') and env.user_persona is not None:
        user_persona = env.user_persona.copy()
        user_persona.pop('__reasoning__', None)
        return_info['user_persona'] = user_persona
    return return_info


def _extract_action_from_response(response: str) -> str:
    """
    Extract action from response

    Args:
        response: Raw response string

    Returns:
        Extracted action string
    """
    matches = re.findall(
        r"(?:^|\n)\s*Action\s*:\s*(.+?)\s*$",
        response,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if len(matches) == 1:
        return matches[0].strip()
    if len(matches) > 1:
        # Preserve the complete response so the strict environment parser can
        # report multiple_actions_not_allowed instead of silently taking one.
        return response.strip()
    return response.strip()


def _handle_interact_action(
    env: Any,
    env_idx: int,
    response: str
) -> Dict[str, Any]:
    """
    Handle interact action

    Args:
        env: Environment object
        env_idx: Environment index
        response: Response string

    Returns:
        Dictionary containing interaction information
    """
    logger.info(
        f"[Interact] Received response: {response}, "
        f"environment index: {env_idx}, Session ID: {env.session}"
    )

    # Normalize response format
    normalized_response = response.replace("\\n", "\n")
    env.history.append({'role': 'assistant', 'content': normalized_response})

    # Extract action
    action_str = _extract_action_from_response(normalized_response)

    # Execute environment step
    _, status, info = env.step(action_str)
    done, reward = status['done'], status['reward']
    observation_state, observation = _canonical_observation(env)

    # Extract status information
    if done:
        reward_detail = status["reward_detail"]
        purchase = status['purchase']
        goal = status['goal']
    else:
        reward_detail = {}
        purchase = {}
        goal = {}

    # Build return information
    return_info = {
        "done": done,
        "reward": reward,
        "observation": observation,
        "message": "Continue interaction",
        "env_idx": env_idx,
        "idx": env.session,
        "reward_detail": reward_detail,
        "purchase": purchase,
        "goal": goal,
        "over": done,
        "observation_state": observation_state,
        "action_feedback": status.get("action_feedback", {}),
    }
    if return_info["action_feedback"] and not return_info["action_feedback"].get(
        "valid", False
    ):
        return_info["message"] = "Invalid action"
    if done:
        return_info["termination_reason"] = status.get(
            "termination_reason", "environment_done"
        )

    return return_info


def _handle_termination(
    env: Any, env_idx: int, termination_reason: str
) -> Dict[str, Any]:
    """Let a runner end an unpurchased episode through the environment."""
    status = env.server.terminate_unpurchased(env.session, termination_reason)
    observation_state, observation = _canonical_observation(env)
    return {
        "done": True,
        "over": True,
        "reward": status["reward"],
        "observation": observation,
        "message": f"Episode terminated: {termination_reason}",
        "env_idx": env_idx,
        "idx": env.session,
        "reward_detail": status["reward_detail"],
        "purchase": status["purchase"],
        "goal": status["goal"],
        "termination_reason": status["termination_reason"],
        "observation_state": observation_state,
    }


def shop_agent(
    env: Any,
    env_idx: int,
    action: str,
    idx: Optional[int] = None,
    response: Optional[str] = None,
    if_persona: bool = False,
    include_private_goal: bool = False,
    interaction_mode: str = "single_turn",
    termination_reason: str = "action_limit",
) -> Dict[str, Any]:
    """
    Main shop agent function that handles environment reset and interaction actions

    Args:
        env: Environment object
        env_idx: Environment index
        action: Action type ("reset" or "interact")
        idx: Task index, only used for reset action
        response: Response string, only used for interact action

    Returns:
        Dictionary containing action processing results

    Raises:
        ValueError: When action is not "reset" or "interact"
    """
    if action == "reset":
        if idx is None:
            raise ValueError("reset action requires idx parameter")
        return _handle_reset_action(
            env,
            env_idx,
            idx,
            if_persona=if_persona,
            include_private_goal=include_private_goal,
            interaction_mode=interaction_mode,
        )

    elif action == "interact":
        if response is None:
            raise ValueError("interact action requires response parameter")
        return _handle_interact_action(env, env_idx, response)

    elif action == "terminate":
        return _handle_termination(env, env_idx, termination_reason)

    else:
        raise ValueError(
            f"Unknown action type: {action}, supported actions: "
            "'reset', 'interact', 'terminate'"
        )
