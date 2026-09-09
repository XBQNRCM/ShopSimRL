import json
import os
import random
import time
import numpy as np

from bs4 import BeautifulSoup
from collections import defaultdict
from flask import Flask
from web_agent_site.engine.engine import (
    ActionParseError,
    load_products,
    init_search_engine,
    get_top_n_product_from_keywords,
    map_action_to_html,
    parse_action,
    get_product_per_page,
    normalize_query,
    ACTION_TO_TEMPLATE,
    PRODUCT_WINDOW,
    SEARCH_RETURN_N,
    END_BUTTON, NEXT_PAGE, PREV_PAGE, BACK_TO_SEARCH,
)
from web_agent_site.engine.options import (
    format_option_action,
    option_selection_state,
)
from web_agent_site.engine.goal import (
    empty_paper_reward_detail,
    get_goals,
    get_reward,
)
from web_agent_site.engine.variant_price import resolve_variant_price
from web_agent_site.engine.observation import (
    build_observation_state,
    page_type_from_name,
    render_observation_state,
)
from web_agent_site.engine.config import (
    ENVIRONMENT_VERSION,
    load_config,
)
from web_agent_site.utils import (BASE_DIR, DEFAULT_FILE_PATH, random_idx)

try:
    import gym
    GymEnv = gym.Env
except ImportError:
    class GymEnv:
        """Small fallback because the HTTP simulator does not require Gym."""

        pass

PROMPT_TEMPLATE_zh="""你正在进行一次网上购物模拟，目标是从商品库中选购最符合需求的商品。请注意，商品库中存在大量同类商品，你必须通过合理操作，最终购买到最符合要求的目标商品。
购物流程采用多轮对话形式，每一轮我会提供当前页面的观察结果，以及你可执行的操作列表。你需要根据当前状态和可选操作，选择并执行最合适的一步。

操作分为两类：
1. 搜索操作
    格式：search[关键词]
    你可以根据当前需求，自主决定搜索关键词。
    只有在搜索功能可用时，才能使用该操作。
2. 点击操作
    格式：click[值]
    你只能点击当前可用操作列表中的按钮或选项，值必须严格对应操作列表中的内容。商品规格使用“轴名=选项值”，例如 click[颜色分类=黑色]。

规则说明：
1. 必须选好商品的全部规格轴后，click[buy now] 才会成为可用动作。
2. 你的目标是综合所有已知信息，通过搜索、筛选和选择，最终购买到“最符合需求的”商品，而非随意购买任意商品。
3. 每轮只能执行一个动作。无效动作不会改变页面，但会消耗一步并返回明确的失败原因。
4. 商品页会明确列出已选择和尚未选择的规格轴；已选择的相同值不会继续作为可用动作。

输出格式
在每一步，你必须按照下面格式输出：
Thought: 简要说明你在当前状态下的思考过程和操作依据。
Action: 用规定格式输出你选择的操作。
"""

PROMPT_TEMPLATE_zh_persona="""你正在进行一次网上购物模拟，目标是从商品库中选购最符合需求的商品。
我会提供他们的目标商品（例如“一双鞋”）以及个人文档（例如偏好、预算、使用场景等）请注意，商品库中存在大量同类商品，你必须通过合理操作，，综合分析用户需求和当前页面信息，最终购买到最符合用户要求的商品。
购物流程采用多轮对话形式，每一轮我会提供当前页面的观察结果，以及你可执行的操作列表。你需要根据当前状态和可选操作，选择并执行最合适的一步。

操作分为两类：
1. 搜索操作
    格式：search[关键词]
    你可以根据当前需求，自主决定搜索关键词。
    只有在搜索功能可用时，才能使用该操作。
2. 点击操作
    格式：click[值]
    你只能点击当前可用操作列表中的按钮或选项，值必须严格对应操作列表中的内容。商品规格使用“轴名=选项值”，例如 click[颜色分类=黑色]。

规则说明：
1. 必须选好商品的全部规格轴后，click[buy now] 才会成为可用动作。
2. 你的目标是综合所有已知信息，通过搜索、筛选和选择，最终购买到“最符合需求的”商品，而非随意购买任意商品。
3. 每轮只能执行一个动作。无效动作不会改变页面，但会消耗一步并返回明确的失败原因。
4. 商品页会明确列出已选择和尚未选择的规格轴；已选择的相同值不会继续作为可用动作。

输出格式
在每一步，你必须按照下面格式输出：
Thought: 简要说明你在当前状态下的思考过程和操作依据。
Action: 用规定格式输出你选择的操作。
"""

app = Flask(__name__)
class WebAgentTextEnv(GymEnv):
    """Gym environment for Text mode of WebShop environment"""
    def __init__(
            self,
            observation_mode='html',
            file_path=DEFAULT_FILE_PATH,
            server=None,
            **kwargs
        ):
        """
        Constructor for text environment

        Arguments:
        observation_mode (`str`) -- ['html' | 'structured_text' | 'url']
        get_image
        filter_goals
        limit_goals
        num_products
        human_goals
        session
        session_prefix
        show_attrs
        """
        super(WebAgentTextEnv, self).__init__()
        self.observation_mode = observation_mode
        self.kwargs = kwargs

        self.file_path = file_path

        self.base_url = 'http://127.0.0.1:3000'
        self._owns_server = server is None
        self.server = SimServer(
            self.base_url,
            self.file_path,
            self.kwargs.get('filter_goals'),
            self.kwargs.get('limit_goals', -1),
            self.kwargs.get('num_products'),
            self.kwargs.get('human_goals'),
            self.kwargs.get('show_attrs', False),
            self.kwargs.get('shuffle_goals', False),
            self.kwargs.get('shuffle_num', 20),
            self.kwargs.get('shift_goals', False),
            self.kwargs.get('if_persona', False),
        ) if server is None else server
        self.browser = SimBrowser(self.server)
        self.session = self.kwargs.get('session')
        self.session_prefix = self.kwargs.get('session_prefix')
        if self.kwargs.get('get_image', 0):
            # TODO: self.ids 需要从外部传入或初始化，否则这里会报错
            # self.ids = {url: idx for idx, url in enumerate(self.ids)}
            pass
        self.prev_obs = []
        self.prev_actions = []
        self.num_prev_obs = self.kwargs.get('num_prev_obs', 0)
        self.num_prev_actions = self.kwargs.get('num_prev_actions', 0)
        self.split = self.kwargs.get('split', "test")
        self.if_persona = bool(self.kwargs.get('if_persona', False))
        if self.if_persona:
            self.prompt_template = PROMPT_TEMPLATE_zh_persona
        else:
            self.prompt_template = PROMPT_TEMPLATE_zh
        self.history_init = [
            {'role': 'user', 'content': self.prompt_template},
            {'role': 'assistant', 'content': 'ok'},
        ]
        if self.split == 'test':
            self.init_idx = -1
        else:
            self.init_idx = 194
        self.idx = self.init_idx

    def step(self, action):
        """
        Takes an action, updates WebShop environment, and returns (observation, reward, done, info)

        Arguments:
        action (`str`): An action should be of the following structure:
          - search[keywords]
          - click[value]
        If action not valid, perform nothing.
        """
        info = None
        session = self.server.user_sessions.get(self.session)
        if session and session.get("done"):
            raise RuntimeError("episode is already terminated; reset before stepping")
        available_actions = self.get_available_actions()
        status = dict(reward=0, done=False)
        feedback = {
            "action": str(action),
            "valid": False,
            "reason": "invalid_format",
            "message": "动作格式无效",
        }

        # Determine and validate exactly one public action.
        try:
            action_name, action_arg = parse_action(action)
        except ActionParseError as exc:
            action_name, action_arg = "", None
            feedback.update(reason=exc.code, message=str(exc))
        if session is not None:
            session["action_count"] += 1

        action_name = action_name.casefold()
        action_key = action_arg if action_arg is not None else None
        selected_action_keys = {
            format_option_action(axis, value)
            for axis, value in (session.get("options") or {}).items()
        } if session is not None else set()

        if action_name == "search" and not (action_arg or "").strip():
            feedback.update(
                reason="empty_argument",
                message="search[...] 的关键词不能为空",
            )
        elif action_name == "search" and not available_actions["has_search_bar"]:
            feedback.update(
                reason="search_unavailable",
                message="当前页面不提供搜索操作",
            )
        elif action_name == "search":
            status = self.browser.search(action_arg)
            feedback.update(valid=True, reason="executed", message="搜索已执行")
        elif action_name == "click" and not (action_arg or "").strip():
            feedback.update(
                reason="empty_argument",
                message="click[...] 的点击值不能为空",
            )
        elif action_name == "click" and action_key in selected_action_keys:
            feedback.update(
                reason="already_selected",
                message=f"该规格已经选中：{action_arg}",
            )
        elif (
            action_name == "click"
            and action_key == END_BUTTON.casefold()
            and session is not None
            and session.get("asin") in self.server.product_item_dict
        ):
            selection = option_selection_state(
                self.server.product_item_dict[session["asin"]],
                session.get("options"),
            )
            if (
                selection["options_complete"]
                and action_key in self.text_to_clickable
            ):
                status = self.browser.click(action_key, self.text_to_clickable)
                feedback.update(
                    valid=True, reason="executed", message="购买已执行"
                )
            elif selection["options_complete"]:
                feedback.update(
                    reason="unavailable_action",
                    message="当前页面没有可用的购买按钮",
                )
            else:
                feedback.update(
                    reason="incomplete_options",
                    message=(
                        "购买前还需选择规格轴："
                        + "、".join(selection["missing_option_axes"])
                    ),
                    missing_option_axes=selection["missing_option_axes"],
                )
        elif (
            action_name == "click"
            and action_key in self.text_to_clickable
            and action_key != "search"
        ):
            status = self.browser.click(action_key, self.text_to_clickable)
            feedback.update(valid=True, reason="executed", message="点击已执行")
        elif action_name == "click":
            feedback.update(
                reason="unavailable_action",
                message=f"当前页面不存在可点击项：{action_arg}",
            )
        elif action_name:
            feedback.update(
                reason="invalid_format",
                message="动作必须使用 search[...] 或 click[...] 格式",
            )

        # Update observation, state with the new action
        if session is not None:
            session["last_action"] = feedback
        ob = self.observation
        if (
            not status.get("done")
            and session is not None
            and session["action_count"] >= session["max_actions"]
        ):
            status = self.server.terminate_action_limit(self.session)
        status["action_feedback"] = feedback
        if not feedback["valid"] and self.observation_mode != "structured_text":
            ob = f"上次操作无效：{feedback['message']}\n\n{ob}"
        text_list = [ob]
        self.prev_actions.append(action)
        for i in range(1, 1 + max(self.num_prev_obs, self.num_prev_actions)):
            if len(self.prev_actions) >= i and self.num_prev_actions >= i:
                text_list.append(self.prev_actions[-i])
            if len(self.prev_obs) >= i and self.num_prev_obs >= i:
                text_list.append(self.prev_obs[-i])
        state = ' [SEP] '.join(text_list[::-1])
        self.prev_obs.append(ob)

        #return state, status['reward'], status['done'], info
        return state, status, info

    def get_available_actions(self):
        """Returns list of available actions at the current step"""
        html_obj = self._parse_html()

        # Collect search bar, buttons, links, and options as clickables
        search_bar = html_obj.find(id='search_input')
        has_search_bar = True if search_bar is not None else False
        buttons = html_obj.find_all(class_='btn')
        product_links  = html_obj.find_all(class_='product-link')
        buying_options = html_obj.select('input[type="radio"]')

        self.text_to_clickable = {}
        clickable_labels = {}
        for clickable in buttons + product_links:
            label = clickable.get_text(strip=True).casefold()
            if not label or label == "search":
                continue
            key = label
            self.text_to_clickable[key] = clickable
            clickable_labels[key] = label

        selected_options = {}
        session = self.server.user_sessions.get(self.session)
        if session is not None:
            selected_options = session.get("options") or {}
        for opt in buying_options:
            axis = str(opt.get("name") or "")
            value = str(opt.get("value") or "")
            label = format_option_action(axis, value)
            if not axis or not value or selected_options.get(axis) == value:
                continue
            key = label
            self.text_to_clickable[key] = opt
            clickable_labels[key] = label
        return dict(
            has_search_bar=has_search_bar,
            clickables=[
                clickable_labels[key]
                for key in self.text_to_clickable
            ],
        )

    def get_image(self):
        """Scrape image from page HTML and return as a list of pixel values"""
        import torch

        html_obj = self._parse_html(self.browser.page_source)
        image_url = html_obj.find(id='product-image')
        if image_url is not None:
            image_url = image_url['src']
            if image_url in self.ids:
                image_idx = self.ids[image_url]
                image = self.feats[image_idx]
                return image
        return torch.zeros(512)

    def get_instruction_text(self):
        """Get corresponding instruction text for current environment session"""
        html_obj = self._parse_html(self.browser.page_source)
        instruction_text = html_obj.find(id='instruction-text').h4.text
        return instruction_text

    def _parse_html(self, html=None):
        """
        Returns web request result wrapped in BeautifulSoup object

        Arguments:
        url (`str`): If no url or html is provided, use the current
            observation (HTML) for parsing.
        """
        if html is None:
            html = self.state['html']
        html_obj = BeautifulSoup(html, 'html.parser')
        return html_obj

    @property
    def observation(self):
        """Compile the environment into the configured observation mode."""
        if self.observation_mode == 'structured_text':
            return render_observation_state(self.structured_observation())
        html = self.state['html']
        if self.observation_mode == 'html':
            return html
        elif self.observation_mode == 'url':
            return self.state['url']
        else:
            raise ValueError(
                f'Observation mode {self.observation_mode} not supported.'
            )

    @property
    def state(self):
        """
        State that includes all information. The actual observation are
        likely to be a subset or reduced form of the state.
        """
        return dict(
            url=self.browser.current_url,
            html=self.browser.page_source,
            instruction_text=self.instruction_text,
        )

    def reset(
        self,
        idx=None,
        session=None,
        instruction_text=None,
        if_persona=None,
        interaction_mode="single_turn",
    ):
        """Create a new session and reset environment variables"""
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise ValueError("task idx must be an integer")
        if idx not in self.server.goals_by_task_id:
            raise KeyError(f"unknown or inactive task_id {idx}")
        session_int = None

        goal = self.server.goals_by_task_id[idx]
        if interaction_mode not in self.server.episode_limits:
            raise ValueError(
                f"unsupported interaction_mode {interaction_mode!r}; "
                f"choose one of {sorted(self.server.episode_limits)}"
            )
        self.if_persona = (
            self.if_persona if if_persona is None else bool(if_persona)
        )
        visible_instruction = instruction_text
        if visible_instruction is None:
            visible_instruction = (
                goal['instruction_simple']
                if self.if_persona and goal.get('user_persona')
                else goal['instruction_text']
            )

        if session is not None:
            self.session = str(session)
            session_int = idx
        elif self.session_prefix is not None:
            self.session = f"{self.session_prefix}-{idx}"
            session_int = idx
        else:
            self.session = idx

        init_url = f'{self.base_url}/{self.session}'
        self.browser.get(
            init_url,
            session_id=self.session,
            session_int=session_int,
            visible_instruction_text=visible_instruction,
            reset_session=True,
        )
        runtime_session = self.server.user_sessions[self.session]
        runtime_session["interaction_mode"] = interaction_mode
        runtime_session["max_actions"] = self.server.episode_limits[interaction_mode]

        self.text_to_clickable = None
        self.instruction_text = visible_instruction
        obs = self.observation
        self.prev_obs = [obs]
        self.prev_actions = []
        self.prompt_template = (
            PROMPT_TEMPLATE_zh_persona if self.if_persona else PROMPT_TEMPLATE_zh
        )
        self.history = [
            {'role': 'user', 'content': self.prompt_template},
            {'role': 'assistant', 'content': 'ok'},
        ]
        self.instruction_simple = goal['instruction_simple']
        self.goal_options = goal['goal_options']
        self.user_persona = goal.get('user_persona', {}).copy()
        self.reason_key = goal.get('reason_key')
        if self.if_persona:
            user_persona = self.user_persona.copy()
            if '__reasoning__' in user_persona:
                del user_persona['__reasoning__']
            self.history.extend([
                {'role': 'user', 'content': f"\n用户个人文档：{json.dumps(user_persona, ensure_ascii=False)}"},
                {'role': 'assistant', 'content': 'ok'},
            ])
        return obs, None

    def structured_observation(self):
        available_actions = self.get_available_actions()
        page_name = self.server.get_page_name(self.browser.current_url)
        return build_observation_state(
            page_type=page_type_from_name(page_name),
            session=self.server.user_sessions[self.session],
            product_item_dict=self.server.product_item_dict,
            available_actions=available_actions,
        )

    def render(self, mode='human'):
        pass

    def close(self):
        if self._owns_server:
            self.server.close()

    def clear_session(self):
        """Discard rollout state before its worker slot becomes reusable."""
        if self.session is not None:
            self.server.user_sessions.pop(self.session, None)
        self.browser.current_url = None
        self.browser.page_source = None
        self.browser.session_id = None
        self.history = []

class SimServer:
    """Lightweight simulator of WebShop Flask application for generating HTML observations"""
    def __init__(
        self,
        base_url,
        file_path,
        filter_goals=None,
        limit_goals=-1,
        num_products=None,
        human_goals=0,
        show_attrs=False,
        shuffle_goals=False,
        shuffle_num=20,
        shift_goals=False,
        if_persona=False,
    ):
        """
        Constructor for simulated server serving WebShop application

        Arguments:
        filter_goals (`func`) -- Select specific goal(s) for consideration based on criteria of custom function
        limit_goals (`int`) -- Limit to number of goals available
        num_products (`int`) -- Number of products to search across
        human_goals (`bool`) -- If true, load human goals; otherwise, load synthetic goals
        """
        self.environment_version = ENVIRONMENT_VERSION
        config_path = os.environ.get(
            "SHOP_ENV_CONFIG",
            os.path.join(BASE_DIR, "../configs/environment.json"),
        )
        self.environment_config = load_config(config_path)

        # Load all products, goals, and search engine
        self.base_url = base_url
        self.all_products, self.product_item_dict, self.product_prices, _ = \
            load_products(filepath=file_path, num_products=num_products, human_goals=human_goals)
        self.search_engine = init_search_engine(
            num_products=num_products,
            product_filepath=file_path,
        )
        search_config = self.environment_config["search"]
        if int(search_config["top_k"]) != SEARCH_RETURN_N:
            raise ValueError("search top_k differs from the engine runtime")
        if int(search_config["page_size"]) != PRODUCT_WINDOW:
            raise ValueError("search page_size differs from the engine runtime")
        if (
            self.search_engine.manifest.get("field_weights")
            != search_config["field_weights"]
        ):
            raise ValueError("search index weights differ from the config")
        self.existed_goals = True
        self.goals = get_goals(self.all_products, self.product_prices, if_persona=if_persona)
        self.show_attrs = show_attrs
        self.shuffle_goals = shuffle_goals
        self.shuffle_num = shuffle_num
        self.shift_goals = shift_goals
        # Fix outcome for random shuffling of goals
        #random.seed(233)
        #random.shuffle(self.goals)

        # Apply `filter_goals` parameter if exists to select speific goal(s)
        if filter_goals is not None:
            self.goals = [
                goal for goal in self.goals
                if filter_goals(goal["task_id"], goal)
            ]

        # Imposes `limit` on goals via random selection
        if limit_goals != -1 and limit_goals < len(self.goals):
            self.weights = [goal['weight'] for goal in self.goals]
            self.cum_weights = [0] + np.cumsum(self.weights).tolist()
            idxs = []
            while len(idxs) < limit_goals:
                idx = random_idx(self.cum_weights)
                if idx not in idxs:
                    idxs.append(idx)
            self.goals = [self.goals[i] for i in idxs]
        self.goals_by_task_id = {}
        for goal in self.goals:
            task_id = int(goal["task_id"])
            if task_id in self.goals_by_task_id:
                raise ValueError(f"duplicate task_id in active goals: {task_id}")
            self.goals_by_task_id[task_id] = goal
        print(f'Loaded {len(self.goals)} goals.')
        # Set extraneous housekeeping variables
        self.user_sessions = dict()
        self.render_time = 0
        self.search_time = 0
        self.sample = 0
        self.assigned_instruction_text = None
        self.episode_limits = {
            name: int(value)
            for name, value in self.environment_config["episode_limits"].items()
        }

    @app.route('/', methods=['GET', 'POST'])
    def index(self, session_id, **kwargs):
        """Redirect to the search page with the given session ID"""
        html = map_action_to_html(
            'start',
            session_id=session_id,
            instruction_text=kwargs['instruction_text'],
        )
        url = f'{self.base_url}/{session_id}'
        return html, url

    @app.route('/', methods=['GET', 'POST'])
    def search_results(self, session_id, **kwargs):
        """Initialize session and return the search results page"""
        session = self.user_sessions[session_id]
        keywords = kwargs['keywords']
        assert isinstance(keywords, list)
        requested_page = kwargs.get("page")
        page = 1 if requested_page is None else requested_page
        if page < 1:
            raise ValueError("search result page must be positive")
        normalized_query = normalize_query(' '.join(keywords))
        is_new_query = (
            requested_page is None
            or session.get("normalized_query") != normalized_query
            or "search_result_asins" not in session
        )
        session["page"] = page
        session["keywords"] = keywords
        session["asin"] = None
        session["options"] = {}

        if is_new_query:
            session["actions"]["search"] += 1
            session["normalized_query"] = normalized_query
            session.setdefault("distinct_normalized_queries", set()).add(
                normalized_query
            )
            old_time = time.time()
            top_n_products = get_top_n_product_from_keywords(
                keywords,
                self.search_engine,
                self.all_products,
                self.product_item_dict,
            )
            self.search_time += time.time() - old_time

            if self.shuffle_goals and self.shuffle_num != 0:
                if len(top_n_products) > self.shuffle_num:
                    first = top_n_products[:self.shuffle_num]
                    random.shuffle(first)
                    top_n_products = first + top_n_products[self.shuffle_num:]
                else:
                    random.shuffle(top_n_products)
            if self.shift_goals and len(top_n_products) > 40:
                top_n_products = (
                    top_n_products[20:40]
                    + top_n_products[:20]
                    + top_n_products[40:]
                )
            session["search_result_asins"] = [
                product["asin"] for product in top_n_products
            ]
        else:
            top_n_products = [
                self.product_item_dict[asin]
                for asin in session["search_result_asins"]
                if asin in self.product_item_dict
            ]

        total_pages = max(
            1, (len(top_n_products) + PRODUCT_WINDOW - 1) // PRODUCT_WINDOW
        )
        if page > total_pages:
            raise ValueError(
                f"search result page {page} is beyond the final page {total_pages}"
            )

        products = get_product_per_page(top_n_products, page)
        session["current_page_asins"] = [
            product["asin"] for product in products
        ]
        session["total_results"] = len(top_n_products)
        session["total_pages"] = total_pages

        keywords_url_string = '+'.join(keywords)
        url = (
            f'{self.base_url}/search_results/{session_id}/'
            f'{keywords_url_string}/{page}'
        )

        # Render HTML search page and record amount of time taken
        old_time = time.time()
        html = map_action_to_html(
            'search',
            session_id=session_id,
            products=products,
            keywords=session["keywords"],
            page=page,
            total=len(top_n_products),
            total_pages=total_pages,
            normalized_query=normalized_query,
            instruction_text=session["visible_instruction_text"],
        )
        self.render_time += time.time() - old_time
        return html, url

    @app.route('/', methods=['GET', 'POST'])
    def item_page(self, session_id, **kwargs):
        """Render and return the HTML for a product item page"""
        session = self.user_sessions[session_id]
        clickable_name = kwargs['clickable_name']
        text_to_clickable = kwargs['text_to_clickable']
        clickable = text_to_clickable[clickable_name]

        # Update session logs with information of last product asin selected
        if (clickable.get('class') is not None and
            clickable.get('class')[0] == 'product-link'):
            session["asin"] = clickable_name.upper()
            session["actions"]["asin"] += 1
            session["asins"].add(session["asin"])
        elif clickable.get('name') is not None:
            clickable_key = str(clickable['name'])
            session["options"][clickable_key] = str(clickable.get('value'))
            session["actions"]["options"] += 1

        # Set fields + url of page, then render page's HTML
        product_info = self.product_item_dict[session["asin"]]
        keywords_url_string = '+'.join(session["keywords"])
        option_string = json.dumps(session['options'])

        # Resolve the selected SKU price from the current option state.
        price_resolution = resolve_variant_price(
            product_info,
            session["options"],
        )
        selected_price = (
            price_resolution["price"]
            if price_resolution["status"] == "pass"
            else None
        )
        session["price_resolution"] = price_resolution
        session["selected_price"] = selected_price
        session["subpage"] = None
        selection = option_selection_state(product_info, session["options"])

        url = (
            f'{self.base_url}/item_page/{session_id}/'
            f'{session["asin"]}/{keywords_url_string}/'
            f'{session["page"]}/{option_string}'
        )

        html = map_action_to_html(
            'click',
            session_id=session_id,
            product_info=product_info,
            keywords=session["keywords"],
            page=session["page"],
            asin=session["asin"],
            options=session["options"],
            instruction_text=session["visible_instruction_text"],
            show_attrs=self.show_attrs,
            selected_price=selected_price,
            missing_option_axes=selection["missing_option_axes"],
            options_complete=selection["options_complete"],
        )
        return html, url

    @app.route('/', methods=['GET', 'POST'])
    def item_sub_page(self, session_id, **kwargs):
        """Render and return the optional product-attributes page."""
        session = self.user_sessions[session_id]
        clickable_name = kwargs['clickable_name']
        for k in ACTION_TO_TEMPLATE:
            if clickable_name.lower() == k.lower():
                clickable_name = k
                break

        # Set fields + url of page, then render page's HTML
        product_info = self.product_item_dict[session["asin"]]
        session["subpage"] = clickable_name
        session["actions"][clickable_name] += 1
        keywords_url_string = '+'.join(session["keywords"])
        url = (
            f'{self.base_url}/item_sub_page/{session_id}/'
            f'{session["asin"]}/{keywords_url_string}/{session["page"]}/'
            f'{clickable_name}/{session["options"]}'
        )
        html = map_action_to_html(
            f'click[{clickable_name}]',
            session_id=session_id,
            product_info=product_info,
            keywords=session["keywords"],
            page=session["page"],
            asin=session["asin"],
            options=session["options"],
            instruction_text=session["visible_instruction_text"],
        )
        return html, url

    @app.route('/', methods=['GET', 'POST'])
    def done(self, session_id, **kwargs):
        """Render and return HTML for done page"""
        session = self.user_sessions[session_id]
        goal = self.user_sessions[session_id]['goal']
        purchased_product = self.product_item_dict[session["asin"]]
        session["actions"]["purchase"] += 1
        price = session.get("selected_price")

        # Calculate reward for selected product and set variables for page details
        reward, info = get_reward(
            purchased_product,
            goal,
            options=session["options"],
            price_resolution=session.get("price_resolution"),
            verbose=True,
        )

        self.user_sessions[session_id]['verbose_info'] = info
        self.user_sessions[session_id]['done'] = True
        self.user_sessions[session_id]['reward'] = reward
        self.user_sessions[session_id]['termination_reason'] = "purchase"

        url = (
            f'{self.base_url}/done/{session_id}/'
            f'{session["asin"]}/{session["options"]}'
        )
        html = map_action_to_html(
            f'click[{END_BUTTON}]',
            session_id=session_id,
            reward=reward,
            asin=session["asin"],
            options=session["options"],
            instruction_text=session["visible_instruction_text"],
        )
        purchase_record = dict(purchased_product)
        purchase_record["price"] = price
        return html, url, reward, info, purchase_record, goal, session["options"]

    def terminate_unpurchased(self, session_id, termination_reason):
        """End an unpurchased episode with zero paper reward."""
        if termination_reason not in {"action_limit", "generation_length"}:
            raise ValueError(
                f"unsupported termination reason: {termination_reason}"
            )
        session = self.user_sessions[session_id]
        if session.get("done"):
            raise RuntimeError("episode is already terminated")
        info = empty_paper_reward_detail(termination_reason=termination_reason)
        session["verbose_info"] = info
        session["done"] = True
        session["reward"] = 0.0
        session["termination_reason"] = termination_reason
        return {
            "reward": 0.0,
            "reward_detail": info,
            "done": True,
            "purchase": {},
            "goal": session["goal"],
            "termination_reason": termination_reason,
        }

    def terminate_action_limit(self, session_id):
        """End an unpurchased episode at the paper's action limit."""
        return SimServer.terminate_unpurchased(self, session_id, "action_limit")

    def receive(
        self,
        session_id,
        current_url,
        session_int=None,
        visible_instruction_text=None,
        reset_session=False,
        **kwargs,
    ):
        """Map action to the corresponding page"""
        status = dict(reward=0.0, done=False)
        with app.app_context(), app.test_request_context():
            # Create/determine goal, instruction_text from current session
            if reset_session or session_id not in self.user_sessions:
                idx = int(session_int) if session_int is not None else int(session_id)
                # Each slot owns its session state, while the immutable goal
                # corpus is shared.  A shallow copy prevents legacy overrides
                # from contaminating another rollout.
                try:
                    goal = dict(self.goals_by_task_id[idx])
                except KeyError as exc:
                    raise KeyError(f"unknown or inactive task_id {idx}") from exc
                instruction_text = (
                    visible_instruction_text
                    if visible_instruction_text is not None
                    else goal['instruction_text']
                )
                self.user_sessions[session_id] = {
                    'goal': goal,
                    'done': False,
                    'visible_instruction_text': instruction_text,
                }
            else:
                instruction_text = self.user_sessions[session_id][
                    'visible_instruction_text'
                ]
            if self.assigned_instruction_text is not None:
                instruction_text = self.assigned_instruction_text
                self.user_sessions[session_id][
                    'visible_instruction_text'
                ] = instruction_text
            session = self.user_sessions[session_id]

            if not kwargs:
                # If no action, reset the session variables
                kwargs['instruction_text'] = instruction_text
                html, url = self.index(session_id, **kwargs)
                self.user_sessions[session_id].update(
                    {
                        'keywords': None,
                        'page': None,
                        'asin': None,
                        'asins': set(),
                        'distinct_normalized_queries': set(),
                        'normalized_query': None,
                        'search_result_asins': [],
                        'current_page_asins': [],
                        'total_results': 0,
                        'total_pages': 0,
                        'options': dict(),
                        'selected_price': None,
                        'subpage': None,
                        'price_resolution': None,
                        'last_action': None,
                        'action_count': 0,
                        'interaction_mode': 'single_turn',
                        'max_actions': self.episode_limits['single_turn'],
                        'actions': defaultdict(int)
                    }
                )
            elif 'keywords' in kwargs:
                # If search keywords are available, run a search
                html, url = self.search_results(session_id, **kwargs)
            elif 'clickable_name' in kwargs:
                clickable_name = kwargs['clickable_name'].lower()
                if clickable_name == END_BUTTON.lower():
                    # If "buy now" clicked, calculate reward and flag session as terminated
                    html, url, reward, reward_detail, purchased_product, goal, options = self.done(session_id, **kwargs)
                    status['reward'] = reward
                    status['reward_detail'] = reward_detail
                    status['done'] = True
                    status['purchase'] = self.get_purchase_info(purchased_product, options)
                    status['goal'] = goal
                    status['termination_reason'] = reward_detail.get(
                        "termination_reason", "environment_done"
                    )
                elif clickable_name == BACK_TO_SEARCH.lower():
                    # Return to the search form without erasing session-level
                    # Search and product history remain available for rendering.
                    html, url = self.index(
                        session_id,
                        instruction_text=instruction_text,
                    )
                    session.update(
                        {
                            "page": None,
                            "asin": None,
                            "current_page_asins": [],
                            "options": {},
                            "selected_price": None,
                            "price_resolution": None,
                            "subpage": None,
                        }
                    )
                elif (clickable_name == NEXT_PAGE.lower() and
                      self.get_page_name(current_url) == 'search_results'):
                    # If "next page" clicked from search results, re-render with `page` enumerated
                    html, url, status = self.receive(
                        session_id,
                        current_url,
                        keywords=session["keywords"],
                        page=session["page"] + 1,
                    )
                elif (clickable_name == PREV_PAGE.lower() and
                      self.get_page_name(current_url) == 'search_results'):
                    # If "prev page" clicked from search results, re-render with `page` denumerated
                    html, url, status = self.receive(
                        session_id,
                        current_url,
                        keywords=session["keywords"],
                        page=session["page"] - 1,
                    )
                elif (clickable_name == PREV_PAGE.lower() and
                      self.get_page_name(current_url) == 'item_sub_page'):
                    # If "prev page" clicked from sub page, return to corresponding item page
                    html, url = self.item_page(session_id, **kwargs)
                elif (clickable_name == PREV_PAGE.lower() and
                      self.get_page_name(current_url) == 'item_page'):
                    # If "prev page" clicked from item page, return to search results page
                    html, url = self.search_results(
                        session_id,
                        keywords=session["keywords"],
                        page=session["page"],
                        **kwargs
                    )
                elif clickable_name in [k.lower() for k in ACTION_TO_TEMPLATE]:
                    # Render the optional product-attributes page.
                    html, url = self.item_sub_page(session_id, **kwargs)
                else:
                    # Otherwise, render current item page
                    html, url = self.item_page(session_id, **kwargs)
            return html, url, status

    def get_purchase_info(self, purchase, option):
        purchase_light = {"asin": purchase["asin"],
                          "category": purchase["category"],
                          "name": purchase["title"],
                          "product_category": purchase["product_category"],
                          "instruction_text": purchase["instruction_text"],
                          "attributes": purchase["Attributes"],
                           "price": purchase["price"],
                           "options":option}
        return purchase_light


    def get_page_name(self, url):
        """Determine which page (i.e. item_page, search_results) the given URL is pointing at"""
        if url is None:
            return None
        page_names = [
            'search_results',
            'item_page',
            'item_sub_page',
            'done'
        ]
        for page_name in page_names:
            if page_name in url:
                return page_name
        return ''  # index page

    def close(self):
        """Release the shared read-only search connection."""
        if self.search_engine is not None:
            self.search_engine.close()
            self.search_engine = None


class SimBrowser:
    """Simulated browser for rendering the HTML source of WebShop environment pages"""
    def __init__(self, server):
        self.server = server
        self.current_url = None
        self.page_source = None
        self.session_id = None

    def get(
        self,
        url,
        session_id=None,
        session_int=None,
        visible_instruction_text=None,
        reset_session=False,
    ):
        """Set browser variables to corresponding link, page HTML for URL"""
        self.session_id = url.split('/')[-1] if session_id is None else session_id
        self.page_source, _, _ = \
            self.server.receive(
                self.session_id,
                self.current_url,
                session_int=session_int,
                visible_instruction_text=visible_instruction_text,
                reset_session=reset_session,
            )
        self.current_url = url

    def click(self, clickable_name, text_to_clickable):
        """Wrapper for `receive` handler for performing click action on current page"""
        self.page_source, self.current_url, status = \
            self.server.receive(
                self.session_id,
                current_url=self.current_url,
                clickable_name=clickable_name,
                text_to_clickable=text_to_clickable,
            )
        return status

    def search(self, keywords):
        """Wrapper for `receive` handler for performing search action on current page"""
        if isinstance(keywords, str):
            keywords = keywords.split(' ')
        self.page_source, self.current_url, status = \
            self.server.receive(
                self.session_id,
                current_url=self.current_url,
                keywords=keywords,
        )
        return status
