import torch

from icg_api.model import EncoderDecoder, RNNDecoder


def test_state_dict_matches_checkpoint_layout(tiny_model):
    keys = tiny_model.state_dict().keys()
    # artifacts/model.pth.tar stores 637 tensors under these names.
    assert len(keys) == 637
    assert {
        "encoder.encoder.0.weight",
        "encoder.embed.weight",
        "decoder.embedding.weight",
        "decoder.lstm.weight_ih_l1",
        "decoder.fcn.weight",
    } <= keys


def test_from_state_dict_infers_dimensions(tiny_model):
    model = EncoderDecoder.from_state_dict(tiny_model.state_dict())
    assert model.encoder.embed.out_features == 8
    assert model.decoder.lstm.hidden_size == 16
    assert model.decoder.lstm.num_layers == 2
    assert model.decoder.fcn.out_features == 7


def test_forward_returns_logits_per_position(tiny_model):
    images = torch.randn(2, 3, 64, 64)
    captions = torch.randint(0, 7, (2, 5))
    assert tiny_model(images, captions).shape == (2, 5, 7)


def _decoder_always_predicting(token: int) -> RNNDecoder:
    decoder = RNNDecoder(embed_size=4, hidden_size=4, vocab_size=5)
    with torch.no_grad():
        decoder.fcn.weight.zero_()
        decoder.fcn.bias.zero_()
        decoder.fcn.bias[token] = 1.0
    return decoder


def test_generate_stops_at_end_token():
    decoder = _decoder_always_predicting(2)
    assert decoder.generate(torch.randn(1, 4), max_length=10, end_index=2) == [2]


def test_generate_respects_max_length():
    decoder = _decoder_always_predicting(4)
    assert decoder.generate(torch.randn(1, 4), max_length=3, end_index=2) == [4, 4, 4]
