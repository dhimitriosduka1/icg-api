import json
import re
import string
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

PAD, SOS, EOS, UNK = "<PAD>", "<SOS>", "<EOS>", "<UNK>"
SPECIAL_TOKENS = (PAD, SOS, EOS, UNK)

_WORDS_WITH_DIGITS = re.compile(r"\S*\d\S*")
_PUNCTUATION = str.maketrans("", "", string.punctuation)


def tokenize(caption: str) -> list[str]:
    """Lowercase, drop words containing digits, strip ASCII punctuation, split on whitespace."""
    caption = _WORDS_WITH_DIGITS.sub("", caption.lower()).strip()
    return caption.translate(_PUNCTUATION).split()


class Vocabulary:
    def __init__(self, tokens: list[str]):
        if tuple(tokens[: len(SPECIAL_TOKENS)]) != SPECIAL_TOKENS:
            raise ValueError(f"Vocabulary must start with {SPECIAL_TOKENS}")
        self.index_to_string = tokens
        self.string_to_index = {token: index for index, token in enumerate(tokens)}

    def __len__(self) -> int:
        return len(self.index_to_string)

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.index_to_string, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def build(cls, captions: Iterable[str], frequency_threshold: int = 5) -> "Vocabulary":
        """Keep words seen at least ``frequency_threshold`` times, in order of reaching it."""
        tokens = list(SPECIAL_TOKENS)
        frequencies: Counter[str] = Counter()
        for caption in captions:
            for word in tokenize(caption):
                frequencies[word] += 1
                if frequencies[word] == frequency_threshold:
                    tokens.append(word)
        return cls(tokens)

    @property
    def end_index(self) -> int:
        return self.string_to_index[EOS]

    def encode(self, caption: str) -> list[int]:
        unknown = self.string_to_index[UNK]
        return [self.string_to_index.get(word, unknown) for word in tokenize(caption)]

    def decode(self, indices: Iterable[int]) -> str:
        """Turn generated indices into a sentence, dropping special tokens."""
        words = []
        for index in indices:
            word = self.index_to_string[index]
            if word == EOS:
                break
            if word not in SPECIAL_TOKENS:
                words.append(word)
        return " ".join(words).capitalize() + "." if words else ""
