from __future__ import annotations

import importlib.util as _importlib_util
from pathlib import Path as _Path


def _load_core_module(module_filename: str):
    here = _Path(__file__).resolve()
    root = here
    while root.name != "stgnn" and root.parent != root:
        root = root.parent
    core_path = root / "core" / module_filename

    spec = _importlib_util.spec_from_file_location(f"stgnn_core_{module_filename}", core_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load core module at {core_path}")
    mod = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = _load_core_module("train_eval_feature_ranking.py")
globals().update({k: v for k, v in _mod.__dict__.items() if not k.startswith("_")})