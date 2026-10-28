import json
import subprocess
import sys

import pytest

from conftest import ROOT
from icg_api.vocabulary import SPECIAL_TOKENS, Vocabulary, tokenize


def test_tokenize_lowercases_and_drops_digits_and_punctuation():
    assert tokenize(" Dy djem, 3 vjeç, luajnë në oborr!") == [
        "dy",
        "djem",
        "vjeç",
        "luajnë",
        "në",
        "oborr",
    ]


def test_build_keeps_words_in_order_of_reaching_threshold():
    vocabulary = Vocabulary.build(["qen burrë", "burrë qen", "burrë mace"], frequency_threshold=2)
    assert vocabulary.index_to_string == [*SPECIAL_TOKENS, "burrë", "qen"]


def test_rejects_vocabulary_without_special_tokens():
    with pytest.raises(ValueError):
        Vocabulary(["një", "burrë"])


def test_encode_maps_unknown_words(vocabulary):
    assert vocabulary.encode("Një burrë, një mace.") == [4, 5, 4, 3]


def test_decode_skips_special_tokens_and_stops_at_eos(vocabulary):
    assert vocabulary.decode([1, 4, 3, 5, 2, 6]) == "Një burrë."


def test_decode_without_words_is_empty(vocabulary):
    assert vocabulary.decode([1, 2]) == ""


def test_save_load_roundtrip(vocabulary, tmp_path):
    vocabulary.save(tmp_path / "vocab.json")
    assert Vocabulary.load(tmp_path / "vocab.json").index_to_string == vocabulary.index_to_string


def test_vocab_artifact_is_reproducible_from_corpus(tmp_path):
    output = tmp_path / "vocab.json"
    subprocess.run(
        [sys.executable, "scripts/build_vocab.py", "--output", str(output)], cwd=ROOT, check=True
    )
    rebuilt = json.loads(output.read_text(encoding="utf-8"))
    shipped = json.loads((ROOT / "artifacts/vocab.json").read_text(encoding="utf-8"))
    assert rebuilt == shipped
    assert len(shipped) == 10078  # Rows in the checkpoint's embedding and output layers.
