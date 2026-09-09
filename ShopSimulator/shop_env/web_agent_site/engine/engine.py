"""
"""
import os
import re
import json
import random
from functools import lru_cache
from collections import defaultdict
from ast import literal_eval
from tqdm import tqdm
from flask import render_template_string
from rich import print
from web_agent_site.utils import BASE_DIR
from web_agent_site.engine.search import (
    MultiFieldBM25Searcher,
    normalize_query,
    sha256_file,
)

TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

SEARCH_RETURN_N = 150
PRODUCT_WINDOW = 20
TOP_K_ATTR = 10

END_BUTTON = 'Buy Now'
NEXT_PAGE = 'Next >'
PREV_PAGE = '< Prev'
BACK_TO_SEARCH = 'Back to Search'

ACTION_TO_TEMPLATE = {
    'Attributes': 'attributes_page.html',
}


class ActionParseError(ValueError):
    """A public action failed the one-action string contract."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code

def map_action_to_html(action, **kwargs):
    action_name, action_arg = parse_action(action)
    if action_name == 'start':
        path = os.path.join(TEMPLATE_DIR, 'search_page.html')
        html = render_template_string(
            read_html_template(path=path),
            session_id=kwargs['session_id'],
            instruction_text=kwargs['instruction_text'],
        )
    elif action_name == 'search':
        path = os.path.join(TEMPLATE_DIR, 'results_page.html')
        html = render_template_string(
            read_html_template(path=path),
            session_id=kwargs['session_id'],
            products=kwargs['products'],
            keywords=kwargs['keywords'],
            page=kwargs['page'],
            total=kwargs['total'],
            total_pages=kwargs.get('total_pages'),
            normalized_query=kwargs.get('normalized_query'),
            instruction_text=kwargs['instruction_text'],
        )
    elif action_name == 'click' and action_arg == END_BUTTON:
        path = os.path.join(TEMPLATE_DIR, 'done_page.html')
        html = render_template_string(
            read_html_template(path),
            session_id=kwargs['session_id'],
            reward=kwargs['reward'],
            asin=kwargs['asin'],
            options=kwargs['options'],
            reward_info=kwargs.get('reward_info'),
            goal_attrs=kwargs.get('goal_attrs'),
            purchased_attrs=kwargs.get('purchased_attrs'),
            goal=kwargs.get('goal'),
            mturk_code=kwargs.get('mturk_code'),
            category=kwargs.get('category'),
            product_category=kwargs.get('product_category'),
        )
    elif action_name == 'click' and action_arg in ACTION_TO_TEMPLATE:
        path = os.path.join(TEMPLATE_DIR, ACTION_TO_TEMPLATE[action_arg])
        html = render_template_string(
            read_html_template(path),
            session_id=kwargs['session_id'],
            product_info=kwargs['product_info'],
            keywords=kwargs['keywords'],
            page=kwargs['page'],
            asin=kwargs['asin'],
            options=kwargs['options'],
            instruction_text=kwargs.get('instruction_text')
        )
    elif action_name == 'click':
        path = os.path.join(TEMPLATE_DIR, 'item_page.html')
        html = render_template_string(
            read_html_template(path),
            session_id=kwargs['session_id'],
            product_info=kwargs['product_info'],
            keywords=kwargs['keywords'],
            page=kwargs['page'],
            asin=kwargs['asin'],
            options=kwargs['options'],
            instruction_text=kwargs.get('instruction_text'),
            show_attrs=kwargs['show_attrs'],
            selected_option=kwargs.get('selected_option'),
            selected_price=kwargs.get('selected_price'),
            missing_option_axes=kwargs.get('missing_option_axes', []),
            options_complete=bool(kwargs.get('options_complete')),
        )
    else:
        raise ValueError('Action name not recognized.')
    return html


def read_html_template(path):
    with open(path, encoding="utf-8") as f:
        template = f.read()
    return template


def parse_action(action):
    """
    Parse action string to action name and its arguments.
    """
    if not isinstance(action, str):
        raise ActionParseError("invalid_format", "动作必须是字符串")
    text = action.strip()
    if not text:
        raise ActionParseError("invalid_format", "动作不能为空")

    labelled_actions = re.findall(
        r"(?:^|\n)\s*Action\s*:\s*[^\r\n]+",
        text,
        flags=re.IGNORECASE,
    )
    chained_actions = re.search(
        r"\]\s*(?:[,;，；]\s*)?(?:search|click)\s*\[",
        text,
        flags=re.IGNORECASE,
    )
    if len(labelled_actions) > 1 or chained_actions:
        raise ActionParseError(
            "multiple_actions_not_allowed",
            "一次调用只能执行一个 search[...] 或 click[...] 动作",
        )

    if "[" not in text and "]" not in text:
        if "\n" in text or "\r" in text:
            raise ActionParseError("invalid_format", "动作格式无效")
        return text, None

    match = re.fullmatch(r"([^\[\]\r\n]+)\[(.*)\]", text)
    if match is None:
        raise ActionParseError(
            "invalid_format", "动作必须严格使用 search[...] 或 click[...] 格式"
        )
    action_name, action_arg = match.groups()
    action_name = action_name.strip()
    if not action_name:
        raise ActionParseError("invalid_format", "动作名称不能为空")
    return action_name, action_arg


def convert_web_app_string_to_var(name, string):
    if name == 'keywords':
        keywords = string
        if keywords.startswith('['):
            keywords = literal_eval(keywords)
        else:
            keywords = [keywords]
        var = keywords
    elif name == 'page':
        page = string
        page = int(page)
        var = page
    else:
        raise ValueError('Name of variable not recognized.')
    return var


def get_top_n_product_from_keywords(
        keywords,
        search_engine,
        all_products,
        product_item_dict,
        attribute_to_asins=None,
    ):
    if keywords[0] == '<r>':
        top_n_products = random.sample(all_products, k=SEARCH_RETURN_N)
    elif keywords[0] == '<a>':
        attribute = ' '.join(keywords[1:]).strip()
        asins = attribute_to_asins[attribute]
        top_n_products = [p for p in all_products if p['asin'] in asins]
    elif keywords[0] == '<c>':
        category = keywords[1].strip()
        top_n_products = [p for p in all_products if p['category'] == category]
    else:
        keywords = ' '.join(keywords)
        hits = search_engine.search(keywords, k=SEARCH_RETURN_N)
        if hits and hasattr(hits[0], "asin"):
            top_n_asins = [hit.asin for hit in hits]
        else:
            docs = [search_engine.doc(hit.docid) for hit in hits]
            top_n_asins = [json.loads(doc.raw())['id'] for doc in docs]
        top_n_products = [product_item_dict[asin] for asin in top_n_asins if asin in product_item_dict]
    return top_n_products


def get_product_per_page(top_n_products, page):
    if not isinstance(page, int) or page < 1:
        raise ValueError("page must be a positive integer")
    return top_n_products[(page - 1) * PRODUCT_WINDOW:page * PRODUCT_WINDOW]


def generate_product_prices(all_products):
    product_prices = dict()
    for product in all_products:
        asin = product['asin']
        pricing = product['pricing']
        if not pricing:
            price = 100.0
        elif len(pricing) >= 1:
            price = pricing[0]
        product_prices[asin] = price
    return product_prices


@lru_cache(maxsize=4)
def _product_file_sha256(filepath):
    return sha256_file(filepath)


def init_search_engine(num_products=None, product_filepath=None):
    if num_products is not None:
        raise ValueError("the current environment requires the complete frozen product corpus")
    index_path = os.environ.get(
        "SHOP_SEARCH_INDEX",
        os.path.join(BASE_DIR, "../search_engine/products.sqlite3"),
    )
    expected_sha = (
        _product_file_sha256(os.path.abspath(product_filepath))
        if product_filepath
        else None
    )
    return MultiFieldBM25Searcher(
        index_path,
        expected_product_sha256=expected_sha,
    )


def clean_product_keys(products):
    for product in products:
        product.pop('product_information', None)
        product.pop('brand', None)
        product.pop('brand_url', None)
        product.pop('list_price', None)
        product.pop('availability_quantity', None)
        product.pop('availability_status', None)
        product.pop('total_reviews', None)
        product.pop('total_answered_questions', None)
        product.pop('seller_id', None)
        product.pop('seller_name', None)
        product.pop('fulfilled_by_amazon', None)
        product.pop('fast_track_message', None)
        product.pop('aplus_present', None)
        product.pop('small_description_old', None)
    print('Keys cleaned.')
    return products


def load_products(filepath, num_products=None, human_goals=True):
    with open(filepath, encoding="utf-8") as f:
        products = json.load(f)

    print('Products loaded.')
    products = clean_product_keys(products)

    all_reviews = dict()
    all_ratings = dict()

    asins = set()
    all_products = []
    attribute_to_asins = defaultdict(set)
    if num_products is not None:
        products = products[:num_products]

    cnt_1, cnt_2, cnt_3 = 0, 0, 0
    for i, p in tqdm(enumerate(products), total=len(products)):
        asin = p['asin']
        if asin == 'nan' or len(asin) > 20:
            cnt_1 += 1
            continue
        if asin in asins:
            cnt_2 += 1
            continue
        else:
            asins.add(asin)

        products[i]['shop_name'] = p['shop_name']
        products[i]['category'] = p['category']
        products[i]['product_category'] = p.get('product_category', '')
        products[i]['Title'] = p['title']
        products[i]['Description'] = p.get('full_description', '')
        products[i]['Reviews'] = all_reviews.get(asin, [])
        products[i]['Rating'] = all_ratings.get(asin, 'N.A.')

        for r in products[i]['Reviews']:
            if 'score' not in r:
                r['score'] = r.pop('stars')
            if 'review' not in r:
                r['body'] = ''
            else:
                r['body'] = r.pop('review')
        products[i]['BulletPoints'] = p.get('small_description', '') \
            if isinstance(p.get('small_description', ''), list) else [p.get('small_description', '')]

        pricing = p.get('pricing')
        if pricing is None or not pricing:
            pricing = [100.0]
            price_tag = '100.0'
        else:
            for j in range(len(pricing)):
                pricing[j] = pricing[j]

            if len(pricing) == 1:
                price_tag = f"{pricing[0]}"
            else:
                price_tag = f"{pricing[0]} to {pricing[1]}"
                pricing = pricing[:2]
        products[i]['pricing'] = pricing
        products[i]['Price'] = price_tag

        options = dict()
        customization_options = p.get('customization_options', '')
        option_to_image = dict()
        option_to_price = dict()
        if customization_options:
            for option_name, option_contents in customization_options.items():
                if option_contents is None:
                    cnt_3 += 1
                    continue
                option_name = str(option_name)

                option_values = []
                for option_content in option_contents:
                    option_value = option_content['value'].strip().replace('/', ' | ').lower()
                    option_image = option_content.get('image', None)

                    option_values.append(option_value)
                    option_to_image[option_value] = option_image
                    option_to_price[option_value] = option_content.get('price', None)
                options[option_name] = option_values
        products[i]['options'] = options
        products[i]['option_to_image'] = option_to_image
        products[i]['option_to_price'] = option_to_price

        # without color, size, price, availability
        products[i]['Attributes'] = products[i]['attribute']
        products[i]['instruction_text'] = p['instructions'][0]['instruction']
        products[i]['instruction_attributes'] = p['instructions'][0]['attributes']

        products[i]['MainImage'] = p['images'][0]
        products[i]['user_persona'] = p.get('user_persona', None)
        products[i]['reason_key'] = p.get('reason_key', None)

        all_products.append(products[i])

    ins = []
    for p in all_products:
        for product in p['instructions']:
            ins.append(product['instruction'])

    product_item_dict = {p['asin']: p for p in all_products}
    product_prices = generate_product_prices(all_products)
    return all_products, product_item_dict, product_prices, attribute_to_asins
