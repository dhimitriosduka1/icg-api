import io
from pathlib import Path

import pytest
import torch
from PIL import Image

from icg_api.model import EncoderDecoder
from icg_api.vocabulary import SPECIAL_TOKENS, Vocabulary

ROOT = Path(__file__).parents[1]


@pytest.fixture
def vocabulary() -> Vocabulary:
    return Vocabulary([*SPECIAL_TOKENS, "një", "burrë", "qen"])


@pytest.fixture
def tiny_model(vocabulary: Vocabulary) -> EncoderDecoder:
    torch.manual_seed(0)
    return EncoderDecoder(embed_size=8, hidden_size=16, vocab_size=len(vocabulary), num_layers=2)


def image_bytes(mode: str = "RGB", size: tuple[int, int] = (64, 48), fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.effect_noise(size, 64).convert(mode).save(buffer, format=fmt)
    return buffer.getvalue()
