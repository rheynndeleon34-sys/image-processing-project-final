# tests/test_basic_techniques.py

import sys
from pathlib import Path
import pytest

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.image_processor import ImageProcessor

@pytest.fixture
def processor():
    input_dir = Path("input")
    output_dir = Path("output")
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    return ImageProcessor(input_dir, output_dir)

def test_canny_edge(processor):
    success_count, _, _ = processor.process_all_images(['canny_edge'])
    assert success_count > 0

def test_anime_style(processor):
    success_count, _, _ = processor.process_all_images(['anime_style'])
    assert success_count > 0

def test_sepia_tone(processor):
    success_count, _, _ = processor.process_all_images(['sepia_tone'])
    assert success_count > 0

def test_pencil_sketch(processor):
    success_count, _, _ = processor.process_all_images(['pencil_sketch'])
    assert success_count > 0
