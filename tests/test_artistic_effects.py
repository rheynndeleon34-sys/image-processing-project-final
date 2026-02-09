import sys
from pathlib import Path
import pytest  # pyright: ignore[reportMissingImports]

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.image_processor import ImageProcessor


@pytest.fixture
def processor():
    input_dir = Path("input")
    output_dir = Path("output")
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    return ImageProcessor(input_dir, output_dir)


def test_movie_poster(processor):
    success, _, _ = processor.process_all_images(["movie_poster"])
    assert success > 0


def test_album_cover(processor):
    success, _, _ = processor.process_all_images(["album_cover"])
    assert success > 0


def test_vhs_effect(processor):
    success, _, _ = processor.process_all_images(["vhs_effect"])
    assert success > 0


def test_pointillism(processor):
    success, _, _ = processor.process_all_images(["pointillism"])
    assert success > 0


def test_security_camera(processor):
    success, _, _ = processor.process_all_images(["security_camera"])
    assert success > 0


def test_film_burn(processor):
    success, _, _ = processor.process_all_images(["film_burn"])
    assert success > 0


def test_embroidery(processor):
    success, _, _ = processor.process_all_images(["embroidery"])
    assert success > 0
