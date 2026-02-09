# tests/test_main.py
import subprocess
from pathlib import Path
import sys
import os

def test_main_runs_fast():
    """
    Fast smoke test: run main.py with only basic techniques
    Passes if main.py executes without crashing
    """
    input_dir = Path("input")
    input_dir.mkdir(exist_ok=True)

    # Force UTF-8 environment for Windows
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        [sys.executable, "main.py", "--basic"],  # only 8 basic techniques
        capture_output=True, text=True,
        env=env  # use UTF-8
    )

    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0
