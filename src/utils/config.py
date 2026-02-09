"""Minimal YAML config loader with CLI override support."""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a dict to a SimpleNamespace."""
    ns = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, _dict_to_namespace(v))
        else:
            setattr(ns, k, v)
    return ns


def _namespace_to_dict(ns: SimpleNamespace) -> dict:
    """Recursively convert a SimpleNamespace to a dict."""
    d = {}
    for k, v in vars(ns).items():
        if isinstance(v, SimpleNamespace):
            d[k] = _namespace_to_dict(v)
        else:
            d[k] = v
    return d


def _set_nested(d: dict, dotted_key: str, value: str) -> None:
    """Set a value in a nested dict using dot notation (e.g., 'channel.snr_db_train=5')."""
    keys = dotted_key.split(".")
    current = d
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    # Attempt to cast value to the appropriate type
    current[keys[-1]] = _auto_cast(value)


def _auto_cast(value: str) -> Any:
    """Cast a string value to int, float, bool, or leave as str."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def load_config(path: str, overrides: list[str] | None = None) -> SimpleNamespace:
    """Load YAML config and apply CLI overrides.

    Args:
        path: Path to YAML config file.
        overrides: List of 'dotted.key=value' strings. If None, reads from sys.argv.

    Returns:
        SimpleNamespace with config values accessible as attributes.
    """
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    # Apply CLI overrides
    if overrides is None:
        overrides = [arg for arg in sys.argv[1:] if "=" in arg]
    for override in overrides:
        key, value = override.split("=", 1)
        _set_nested(cfg_dict, key, value)

    return _dict_to_namespace(cfg_dict)
