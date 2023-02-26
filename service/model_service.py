import torch
import logging
import torchvision.transforms as transforms

from model.encoder_decoder import EncoderDecoder
from vocabulary.vocabulary import Vocabulary
from PIL import Image


def init_model(config):
    logging.info("Initializing vocabulary")
    vocabulary = Vocabulary()
    logging.info(f"Vocabulary successfully initialized with length {len(vocabulary)}")

    embed_size = config['embed_size']
    hidden_size = config['hidden_size']
    num_layers = config['num_layers']
    vocabulary_size = len(vocabulary)

    logging.info(
        f"Creating model with: embed_size={embed_size}, hidden_size={hidden_size}, num_layers={num_layers}, vocab_size={vocabulary_size}"
    )
    model = EncoderDecoder(
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        vocab_size=vocabulary_size
    )
    logging.info("Loading model")
    saved_state = torch.load("model/model.pth.tar", map_location=torch.device('cpu'))
    model.load_state_dict(saved_state["model_state"])
    logging.info("Successfully loaded model")
    model.eval()
    return model, vocabulary


def init_transform(config):
    transformations = config['transformation']
    std = transformations['std']
    mean = transformations['mean']
    resize = transformations['resize']
    image_size = transformations['image_size']
    transform = transforms.Compose([
        transforms.Resize((resize['w'], resize['h'])),
        transforms.RandomCrop((image_size['w'], image_size['h'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    logging.info(f"Using transformation: {transform}")
    return transform


def predict(model, transform, image, vocabulary):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    pil_image = Image.open(image)
    image = transform(pil_image)
    image = [image.unsqueeze(0)]
    image = torch.cat(image, dim=0)
    features = model.encoder(image[0:1].to(device))
    caption = model.decoder.generate_caption(features.unsqueeze(0), vocab=vocabulary)
    return " ".join(caption[1:-1]).capitalize() + "."
