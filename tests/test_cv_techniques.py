# tests/test_cv_techniques.py

import sys
from pathlib import Path
import pytest # type: ignore

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.image_processor import ImageProcessor

@pytest.fixture
def processor():
    input_dir = Path("input")
    output_dir = Path("output")
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    return ImageProcessor(input_dir, output_dir)

def test_panorama_stitching(processor):
    success_count, _, _ = processor.process_all_images(['image_stitching'])
    assert success_count > 0

def test_background_subtraction(processor):
    success_count, _, _ = processor.process_all_images(['background_subtraction'])
    assert success_count > 0

def test_image_compression(processor):
    success_count, _, _ = processor.process_all_images(['image_compression'])
    assert success_count > 0

def test_style_transfer(processor):
    success_count, _, _ = processor.process_all_images(['style_transfer'])
    assert success_count > 0

def test_optical_flow(processor):
    success_count, _, _ = processor.process_all_images(['optical_flow'])
    assert success_count > 0
