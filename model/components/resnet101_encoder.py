import torch.nn as nn
import torchvision.models as models


class ResNet101Encoder(nn.Module):
    def __init__(self, embed_size):
        super(ResNet101Encoder, self).__init__()
        encoder = models.resnet101(pretrained=True)
        modules = list(encoder.children())[:-1]
        self.encoder = nn.Sequential(*modules)
        self.embed = nn.Linear(encoder.fc.in_features, embed_size)

    def forward(self, images):
        features = self.encoder(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
