from pathlib import Path

# MAIN ROUTE
ROOT_DIR = Path(__file__).resolve().parents[2]   #tennis-analysis/
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
DOCS_DIR = ROOT_DIR / "docs"

# SUBFOLDERS
RAW_DATA_DIR = DATA_DIR / "raw"
ANNOTATIONS_DIR = DATA_DIR / "annotations" / "Tennis-Tracking-Detection.v1-v1.yolov8"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

YOLO_MODELS_DIR = MODELS_DIR / "yolo"
#POSE_MODELS_DIR = MODELS_DIR / "pose"
#TRACKING_MODELS_DIR = MODELS_DIR / "tracking"

#YOLO_RUNS_DIR = EXPERIMENTS_DIR / "yolo_runs"
#TRACKING_RUNS_DIR = EXPERIMENTS_DIR / "tracking_runs"
#POSE_RUNS_DIR = EXPERIMENTS_DIR / "pose_runs"

# IMPORTANT FILES
DATA_YAML = ANNOTATIONS_DIR / "data.yaml"
README = ROOT_DIR / "README.md"
REQUIREMENTS = ROOT_DIR / "requirements.txt"
