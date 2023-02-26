import json
import logging
import os.path
import string
import re
import nltk

import pandas as pd
from nltk.tokenize import word_tokenize

nltk.download('punkt')


class Vocabulary:
    def __init__(self, frequency_threshold=5):
        self.index_to_string = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.string_to_index = {v: k for k, v in self.index_to_string.items()}
        self.frequency_threshold = frequency_threshold
        dataframe = pd.read_csv("corpus/annotations.csv", sep="|")
        self.build_vocab(dataframe["caption"].tolist())

    def __len__(self):
        return len(self.index_to_string)

    def build_vocab(self, sentence_list):

        string_to_index_file = "corpus/string_to_index.json"
        if os.path.isfile(string_to_index_file):
            with open(string_to_index_file) as f:
                self.string_to_index = json.load(f)
                self.string_to_index = {k: int(v) for k, v in self.string_to_index.items()}

        index_to_string_file = "corpus/index_to_string.json"
        if os.path.isfile(index_to_string_file):
            with open(index_to_string_file) as f:
                self.index_to_string = json.load(f)
                self.index_to_string = {int(k): v for k, v in self.index_to_string.items()}

        if len(self.index_to_string) > 4 and len(self.string_to_index) > 4:
            logging.info("Loading vocabulary from saved files")
            return

        frequencies = {}

        index = 4

        for sentence in sentence_list:
            for word in tokenize(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.frequency_threshold:
                    self.string_to_index[word] = index
                    index += 1

        self.index_to_string = {v: k for k, v in self.string_to_index.items()}

        with open(string_to_index_file, 'w') as file:
            file.write(json.dumps(self.string_to_index))

        with open(index_to_string_file, 'w') as file:
            file.write(json.dumps(self.index_to_string))

    def numericalize(self, caption):
        return [self.string_to_index[token] if token in self.string_to_index else self.string_to_index["<UNK>"] for
                token in tokenize(caption)]


def clean_caption(caption):
    caption = caption.lower()  # converting string to lowercase
    pattern = "\S*\d\S*"
    caption = re.sub(pattern, "", caption).strip()  # removing words that contain numbers
    caption = caption.translate(str.maketrans('', '', string.punctuation))  # removing punctuation
    return caption


def tokenize(caption):
    return word_tokenize(clean_caption(caption))
