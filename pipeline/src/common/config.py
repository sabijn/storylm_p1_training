from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a yaml config file into a plain dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
