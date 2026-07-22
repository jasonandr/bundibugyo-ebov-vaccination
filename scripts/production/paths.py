"""Paths for the self-contained production workflow."""
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "data_and_results"
FIGURE_DIR = PROJECT_ROOT / "figures" / "current_review"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def figure_path(filename: str) -> Path:
    return ensure_dir(FIGURE_DIR) / filename


def result_path(filename: str) -> Path:
    return ensure_dir(RESULTS_DIR) / filename
