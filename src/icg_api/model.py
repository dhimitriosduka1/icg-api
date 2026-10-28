"""ResNet-101 encoder + LSTM decoder used to generate captions.

Attribute names in these modules define the checkpoint's state-dict keys, so
renaming any of them breaks loading of ``artifacts/model.pth.tar``.
"""

import re

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet101Encoder(nn.Module):
    """Encodes an image batch into one ``embed_size`` feature vector per image."""

    def __init__(self, embed_size: int, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet101(weights=weights)
        # Drop the ImageNet classification head; keep everything up to global pooling.
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images).flatten(start_dim=1)
        return self.embed(features)


class RNNDecoder(nn.Module):
    """LSTM language model conditioned on the image features as its first input."""

    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fcn = nn.Linear(hidden_size, vocab_size)

    def forward(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """Teacher-forced logits of shape ``(batch, seq_len, vocab_size)`` for training."""
        embeds = self.embedding(captions[:, :-1])
        x = torch.cat((features.unsqueeze(1), embeds), dim=1)
        x, _ = self.lstm(x)
        return self.fcn(x)

    def generate(self, features: torch.Tensor, max_length: int, end_index: int) -> list[int]:
        """Greedily decode token indices for a single image's ``(1, embed_size)`` features.

        Decoding stops after ``end_index`` is produced or ``max_length`` tokens.
        """
        inputs = features.unsqueeze(1)
        hidden = None
        indices: list[int] = []
        for _ in range(max_length):
            output, hidden = self.lstm(inputs, hidden)
            predicted = self.fcn(output.squeeze(1)).argmax(dim=1)
            indices.append(predicted.item())
            if indices[-1] == end_index:
                break
            inputs = self.embedding(predicted).unsqueeze(1)
        return indices


class EncoderDecoder(nn.Module):
    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int, num_layers: int = 1):
        super().__init__()
        self.encoder = ResNet101Encoder(embed_size)
        self.decoder = RNNDecoder(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(images), captions)

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> "EncoderDecoder":
        """Build a model whose dimensions are read from ``state_dict``, then load it."""
        embed_size = state_dict["decoder.embedding.weight"].shape[1]
        vocab_size, hidden_size = state_dict["decoder.fcn.weight"].shape
        num_layers = sum(
            bool(re.fullmatch(r"decoder\.lstm\.weight_ih_l\d+", k)) for k in state_dict
        )
        model = cls(embed_size, hidden_size, vocab_size, num_layers)
        model.load_state_dict(state_dict)
        return model
