# tests/test_artistic_effects.py

import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.image_processor import ImageProcessor

@pytest.fixture
def processor():
    input_dir = Path("input")
    output_dir = Path("output")
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    return ImageProcessor(input_dir, output_dir)

def test_oil_painting(processor):
    success_count, _, _ = processor.process_all_images(['oil_painting'])
    assert success_count > 0

def test_cartoon(processor):
    success_count, _, _ = processor.process_all_images(['cartoon'])
    assert success_count > 0

def test_hdr(processor):
    success_count, _, _ = processor.process_all_images(['hdr'])
    assert success_count > 0

def test_watercolor(processor):
    success_count, _, _ = processor.process_all_images(['watercolor'])
    assert success_count > 0

def test_vignette(processor):
    success_count, _, _ = processor.process_all_images(['vignette'])
    assert success_count > 0
