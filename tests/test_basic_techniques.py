import sys
from pathlib import Path
import pytest # pyright: ignore[reportMissingImports]

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
    success, _, _ = processor.process_all_images(["canny_edge"])
    assert success > 0


def test_anime_style(processor):
    success, _, _ = processor.process_all_images(["anime_style"])
    assert success > 0


def test_sepia_tone(processor):
    success, _, _ = processor.process_all_images(["sepia_tone"])
    assert success > 0


def test_pencil_sketch(processor):
    success, _, _ = processor.process_all_images(["pencil_sketch"])
    assert success > 0


def test_sharpen(processor):
    success, _, _ = processor.process_all_images(["sharpen"])
    assert success > 0


def test_edge_detection(processor):
    success, _, _ = processor.process_all_images(["edge_detection"])
    assert success > 0


def test_binary_threshold(processor):
    success, _, _ = processor.process_all_images(["binary_threshold"])
    assert success > 0


def test_emboss(processor):
    success, _, _ = processor.process_all_images(["emboss"])
    assert success > 0
