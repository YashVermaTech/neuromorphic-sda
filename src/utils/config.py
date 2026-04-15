"""
Configuration management for neuromorphic-sda.

Loads YAML config files, merges with defaults, and provides
a dot-access Config object throughout the pipeline.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class Config:
    """
    Hierarchical configuration object with dot-notation access.

    Wraps a nested dictionary so that ``cfg.snn.lif.beta`` works
    in addition to ``cfg['snn']['lif']['beta']``.

    Parameters
    ----------
    data : dict
        Configuration dictionary (may be nested).
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data: Dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, dict):
                self._data[key] = Config(value)
            else:
                self._data[key] = value

    # ------------------------------------------------------------------
    # Dict-like interface
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{name}'") from None

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, dict):
            self._data[key] = Config(value)
        else:
            self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        """Return value for *key* or *default* if not found."""
        return self._data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Recursively convert back to a plain dictionary."""
        result: Dict[str, Any] = {}
        for key, value in self._data.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"

    # ------------------------------------------------------------------
    # Merging helpers
    # ------------------------------------------------------------------

    def update(self, other: Union["Config", Dict[str, Any]]) -> None:
        """
        Recursively update this config with values from *other*.

        Parameters
        ----------
        other : Config | dict
            Source of overriding values.
        """
        if isinstance(other, Config):
            other = other.to_dict()
        for key, value in other.items():
            if key in self._data and isinstance(self._data[key], Config) and isinstance(value, dict):
                self._data[key].update(value)
            else:
                self[key] = value


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new dict that is *base* recursively overridden by *override*."""
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    *,
    defaults_path: Optional[Union[str, Path]] = None,
) -> Config:
    """
    Load a YAML configuration file, optionally merging with defaults.

    The lookup order for *defaults_path* when not supplied:

    1. ``configs/default_config.yaml`` relative to the project root
       (two levels above this file).
    2. Built-in minimal defaults (no YAML required).

    Parameters
    ----------
    config_path : str | Path | None
        Path to a user-supplied YAML config.  If ``None`` only the
        defaults are loaded.
    defaults_path : str | Path | None
        Explicit path to the defaults YAML.  Overrides auto-discovery.

    Returns
    -------
    Config
        Merged configuration object.

    Examples
    --------
    >>> cfg = load_config()                          # defaults only
    >>> cfg = load_config("configs/my_run.yaml")    # merged with defaults
    """
    # ── locate the defaults ─────────────────────────────────────────────
    if defaults_path is None:
        # This file lives at  src/utils/config.py
        # Project root is two levels up.
        _here = Path(__file__).resolve().parent
        _root = _here.parent.parent
        defaults_path = _root / "configs" / "default_config.yaml"

    base_data: Dict[str, Any] = {}
    if Path(defaults_path).exists():
        with open(defaults_path, "r", encoding="utf-8") as fh:
            base_data = yaml.safe_load(fh) or {}

    # ── load user config ────────────────────────────────────────────────
    user_data: Dict[str, Any] = {}
    if config_path is not None:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as fh:
            user_data = yaml.safe_load(fh) or {}

    merged = _deep_merge(base_data, user_data)
    cfg = Config(merged)

    # ── resolve device ──────────────────────────────────────────────────
    if "project" in cfg and cfg.project.get("device") == "auto":
        cfg.project["device"] = _resolve_device()

    return cfg


def _resolve_device() -> str:
    """Return the best available torch device string."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def save_config(cfg: Config, path: Union[str, Path]) -> None:
    """
    Persist a Config object to a YAML file.

    Parameters
    ----------
    cfg : Config
        Configuration to save.
    path : str | Path
        Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(cfg.to_dict(), fh, default_flow_style=False, sort_keys=False)
