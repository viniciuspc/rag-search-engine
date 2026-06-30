import json
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
GOLDEN_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "golden_dataset.json")

def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r") as f:
        data = json.load(f)
    return data["test_cases"]