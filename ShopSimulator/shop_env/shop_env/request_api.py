"""Minimal ShopSimulator HTTP client and smoke-test example."""
import os
import requests
from typing import Optional, Any


# API endpoint URL
API_URL = os.environ.get(
    "SHOPSIM_API_URL", "http://127.0.0.1:5700/api/shop_agent"
)

def init_shop_env(
    action: str,
    env_idx: Optional[int] = None,
    response: Optional[Any] = None,
    idx: Optional[int] = None,
    if_persona: Optional[bool] = False,
    lease_id: Optional[str] = None,
) -> requests.Response:
    """
    Initialize shop environment via API call.

    Args:
        action: The action to perform (e.g., "reset", "release_all", "release_one")
        env_idx: Optional environment index
        response: Optional response data
        idx: Optional index parameter

    Returns:
        Response object from the API request

    Raises:
        requests.RequestException: If the API request fails
    """
    data = {
        'action': action,
        'env_idx': env_idx,
        'response': response,
        'idx': idx,
        'if_persona': if_persona,
        'lease_id': lease_id,
    }

    try:
        http_response = requests.post(API_URL, json=data, timeout=10)
        http_response.raise_for_status()  # Raise an exception for bad status codes
        return http_response
    except requests.RequestException as e:
        raise requests.RequestException(
            f"Failed to call shop_agent API: {e}"
        ) from e


if __name__ == '__main__':
    try:
        # Test release_all action
        result = init_shop_env("release_all")
        print(result.json()['result'])

        # Test reset action
        print("reset 1")
        result = init_shop_env("reset", idx=2)
        result_data = result.json()['result']
        env_idx = result_data['env_idx']
        lease_id = result_data['lease_id']
        print(result_data)

        # Test release_one action
        result = init_shop_env(
            "release_one", env_idx=env_idx, lease_id=lease_id
        )
        print(result.json()['result'])


        # Test reset with persona
        print("reset 1")
        result = init_shop_env("reset", idx=2, if_persona=True)
        result_data = result.json()['result']
        env_idx = result_data['env_idx']
        lease_id = result_data['lease_id']
        print(result_data)

        # Test release_one action
        result = init_shop_env(
            "release_one", env_idx=env_idx, lease_id=lease_id
        )
        print(result.json()['result'])

    except requests.RequestException as e:
        print(f"Error: {e}")
    except KeyError as e:
        print(f"Error: Missing key in response - {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
