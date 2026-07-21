from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "data_and_results"
FIGURE_DIR = PROJECT_ROOT / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def figure_path(filename):
    return ensure_dir(FIGURE_DIR) / filename


def result_path(filename):
    return ensure_dir(RESULTS_DIR) / filename


def data_path(filename):
    return ensure_dir(DATA_DIR) / filename
