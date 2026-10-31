"""Run every task script in order: python run_all.py"""
import runpy
import sys
from pathlib import Path

TASKS = Path(__file__).resolve().parent / "tasks"
sys.path.insert(0, str(TASKS))

for script in sorted(TASKS.glob("q*.py")):
    print(f"\n{'=' * 70}\n{script.stem}\n{'=' * 70}")
    runpy.run_path(str(script), run_name="__main__")
