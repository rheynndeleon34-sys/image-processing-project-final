import sys
import os

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_main_runs():
    import main   # now Python can find main.py
    assert True
