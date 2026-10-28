import logging
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

from icg_api.model import EncoderDecoder
from icg_api.vocabulary import Vocabulary

logger = logging.getLogger(__name__)

# Preprocessing the model was trained with.
RESIZE = (356, 356)
CROP = (224, 224)
MEAN = (0.485, 0.456, 0.405)
STD = (0.229, 0.224, 0.225)

_LFS_POINTER_PREFIX = b"version https://git-lfs"


class CheckpointError(RuntimeError):
    pass


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    with path.open("rb") as file:
        if file.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX:
            raise CheckpointError(
                f"{path} is a Git LFS pointer, not the model weights. "
                "Install Git LFS and run `git lfs pull` to download them."
            )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint["model_state"]


class Captioner:
    def __init__(
        self,
        model: EncoderDecoder,
        vocabulary: Vocabulary,
        device: torch.device | str = "cpu",
        max_length: int = 20,
    ):
        if model.decoder.fcn.out_features != len(vocabulary):
            raise CheckpointError(
                f"Model predicts {model.decoder.fcn.out_features} tokens "
                f"but the vocabulary has {len(vocabulary)}"
            )
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.transform = T.Compose(
            [T.Resize(RESIZE), T.CenterCrop(CROP), T.ToTensor(), T.Normalize(MEAN, STD)]
        )

    @classmethod
    def from_files(
        cls,
        checkpoint_path: str | Path,
        vocabulary_path: str | Path,
        device: str = "auto",
        max_length: int = 20,
    ) -> "Captioner":
        vocabulary = Vocabulary.load(vocabulary_path)
        logger.info("Loaded vocabulary of %d tokens from %s", len(vocabulary), vocabulary_path)
        model = EncoderDecoder.from_state_dict(load_state_dict(checkpoint_path))
        logger.info("Loaded model weights from %s", checkpoint_path)
        return cls(model, vocabulary, resolve_device(device), max_length)

    @torch.inference_mode()
    def caption(self, image: Image.Image) -> str:
        pixels = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        features = self.model.encoder(pixels)
        indices = self.model.decoder.generate(features, self.max_length, self.vocabulary.end_index)
        return self.vocabulary.decode(indices)
