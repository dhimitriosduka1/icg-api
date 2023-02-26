import torch
import torch.nn as nn


class RNNDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(RNNDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fcn = nn.Linear(hidden_size, vocab_size)
        self.drop = nn.Dropout(0.3)

    def forward(self, features, captions):
        embeds = self.embedding(captions[:, :-1])
        x = torch.cat((features.unsqueeze(1), embeds), dim=1)
        x, _ = self.lstm(x, (h, c))
        x = self.fcn(x)
        return x

    def generate_caption(self, inputs, hidden=None, max_len=20, vocab=None):
        batch_size = inputs.size(0)
        captions = []
        for _ in range(max_len):
            output, hidden = self.lstm(inputs, hidden)
            output = self.fcn(output)
            output = output.view(batch_size, -1)

            predicted_word_idx = output.argmax(dim=1)

            captions.append(predicted_word_idx.item())

            if vocab.index_to_string[predicted_word_idx.item()] == "<EOS>":
                break

            inputs = self.embedding(predicted_word_idx.unsqueeze(0))

        return [vocab.index_to_string[idx] for idx in captions]
