"""Load config/config.yaml and resolve paths relative to the project root."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    path = Path(path) if path else ROOT / "config" / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(rel_path):
    """Turn a config-relative path into an absolute path."""
    return ROOT / rel_path
