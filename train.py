import runpy
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
AI_DIR = ROOT_DIR / "core" / "ai"

sys.path.insert(0, str(AI_DIR))
runpy.run_path(str(AI_DIR / "train.py"), run_name="__main__")
