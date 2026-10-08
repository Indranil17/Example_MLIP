"""Load foundation models from a YAML registry.

Registry entry fields
---------------------
name          short label used in output tables
loader        "mace_mp" (model name known to mace.calculators.mace_mp) or "file"
model         mace_mp model name, e.g. medium-mpa-0
url / path    for loader "file": where to download the .model file from, or a local path
head          optional head for multihead models (MACE-MH-1 uses omat_pbe for PBE-level materials)
licence       MIT or ASL; informational, printed in tables
redistribute  true only for MIT models; nothing derived from an ASL model is committed
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import yaml


def load_registry(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        reg = yaml.safe_load(fh)
    models = reg["models"] if isinstance(reg, dict) and "models" in reg else reg
    for m in models:
        m.setdefault("loader", "mace_mp")
        m.setdefault("head", None)
        m.setdefault("licence", "unknown")
        m.setdefault("redistribute", False)
    return models


def ensure_file(url: str, cache_dir: str | Path = "models_cache") -> Path:
    """Download a model file once into cache_dir; return its path. Prints the SHA-256."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / Path(url).name
    if not target.exists():
        print(f"[models] downloading {url} -> {target}")
        urllib.request.urlretrieve(url, target)  # noqa: S310 (fixed, documented URLs)
    h = hashlib.sha256(target.read_bytes()).hexdigest()
    print(f"[models] {target.name} sha256 {h}")
    return target


def load_calculator(entry: dict, device: str = "cpu", dtype: str = "float64", cache_dir: str | Path = "models_cache"):
    """Return an ASE calculator for a registry entry. Dispersion is always off (stated in READMEs)."""
    head = entry.get("head")
    if entry["loader"] == "mace_mp":
        from mace.calculators import mace_mp

        kwargs = dict(model=entry["model"], device=device, default_dtype=dtype, dispersion=False)
        if head:
            kwargs["head"] = head
        try:
            return mace_mp(**kwargs)
        except TypeError:
            # older mace_mp signatures have no head argument
            kwargs.pop("head", None)
            return mace_mp(**kwargs)
    if entry["loader"] == "file":
        from mace.calculators import MACECalculator

        path = entry.get("path") or ensure_file(entry["url"], cache_dir)
        kwargs = dict(model_paths=str(path), device=device, default_dtype=dtype)
        if head:
            kwargs["head"] = head
        try:
            return MACECalculator(**kwargs)
        except TypeError:
            kwargs.pop("head", None)
            return MACECalculator(**kwargs)
    raise ValueError(f"unknown loader {entry['loader']!r} for model {entry.get('name')!r}")
