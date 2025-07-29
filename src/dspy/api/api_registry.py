"""
Registry for APIs.
"""

from typing import Type
from dspy.api.base import Exchange

API_REGISTRY: dict[str, Type[Exchange]] = {}

def register_api(name):
    """
    Decorator to register an API class under a specified name.

    Parameters:
        name (str): The key under which the API class will be registered.
    
    Returns:
        A decorator that registers the class.
    """
    def decorator(cls):
        API_REGISTRY[name] = cls
        return cls
    return decorator

def get_api(name, **kwargs):
    """
    Factory function to instantiate an API based on its registered name.

    Parameters:
        name (str): The registered name of the API (e.g., "bybit").
        **kwargs: Additional keyword arguments that will be passed to the API's constructor.
    
    Returns:
        An instance of the API class corresponding to the given name.
    
    Raises:
        ValueError: If the API name is not found in the registry.
    """
    if name not in API_REGISTRY:
        available = list(API_REGISTRY.keys())
        raise ValueError(f"API '{name}' is not registered. Available APIs: {available}")
    
    return API_REGISTRY[name](**kwargs)
