import io
import os
from pathlib import Path

import pytest
import torch
from PIL import Image

from conftest import ROOT, image_bytes
from icg_api.captioner import Captioner, CheckpointError, load_state_dict
from icg_api.vocabulary import SPECIAL_TOKENS, Vocabulary


@pytest.fixture
def checkpoint(tiny_model, tmp_path) -> Path:
    path = tmp_path / "model.pth.tar"
    torch.save({"model_state": tiny_model.state_dict()}, path)
    return path


def test_lfs_pointer_gives_helpful_error(tmp_path):
    pointer = tmp_path / "model.pth.tar"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n")
    with pytest.raises(CheckpointError, match="git lfs pull"):
        load_state_dict(pointer)


def test_rejects_vocabulary_that_does_not_match_model(tiny_model):
    with pytest.raises(CheckpointError, match="vocabulary has 5"):
        Captioner(tiny_model, Vocabulary([*SPECIAL_TOKENS, "një"]))


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P"])
def test_captions_any_image_mode(checkpoint, vocabulary, tmp_path, mode):
    vocabulary.save(tmp_path / "vocab.json")
    captioner = Captioner.from_files(checkpoint, tmp_path / "vocab.json", device="cpu")
    image = Image.open(io.BytesIO(image_bytes(mode)))
    assert isinstance(captioner.caption(image), str)


def test_caption_is_deterministic(checkpoint, vocabulary, tmp_path):
    vocabulary.save(tmp_path / "vocab.json")
    captioner = Captioner.from_files(checkpoint, tmp_path / "vocab.json", device="cpu")
    image = Image.effect_noise((300, 200), 64).convert("RGB")
    assert captioner.caption(image) == captioner.caption(image)


REAL_CHECKPOINT = Path(os.environ.get("ICG_CHECKPOINT_PATH", ROOT / "artifacts/model.pth.tar"))


@pytest.mark.skipif(
    REAL_CHECKPOINT.stat().st_size < 1024, reason="real model weights not downloaded (git lfs pull)"
)
def test_real_checkpoint_produces_albanian_caption():
    captioner = Captioner.from_files(REAL_CHECKPOINT, ROOT / "artifacts/vocab.json", device="cpu")
    sample = Image.open(ROOT / "samples/best_2.png")
    photo = sample.crop((0, sample.height // 5, sample.width, sample.height))
    caption = captioner.caption(photo)
    assert caption.startswith("Një") and caption.endswith(".")
