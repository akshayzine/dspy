"""
Registry for datasets.
"""

from typing import Type

DATASET_REGISTRY: dict[str, Type] = {}

def register_dataset(name):
    """
    Decorator to register a dataset class under a specified name.

    Parameters:
        name (str): The key under which the dataset class will be registered.
    
    Returns:
        A decorator that registers the class.
    """
    def decorator(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator

def get_dataset(name, **kwargs):
    """
    Factory function to instantiate a dataset based on its registered name.

    Parameters:
        name (str): The registered name of the dataset (e.g., "as-733").
        **kwargs: Additional keyword arguments that will be passed to the dataset's constructor.
    
    Returns:
        An instance of the dataset class corresponding to the given name.
    
    Raises:
        ValueError: If the dataset name is not found in the registry.
    """
    if name not in DATASET_REGISTRY:
        available = list(DATASET_REGISTRY.keys())
        raise ValueError(f"Dataset '{name}' is not registered. Available datasets: {available}")
    
    return DATASET_REGISTRY[name](**kwargs)
