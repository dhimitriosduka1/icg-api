# ICG-API: A simple API for generating an image caption in the Albanian language

## Introduction

ICG-API is a simple abstraction of a model built to generate image captions in the Albanian language.
The model leverages an existing dataset with English captions and employs a pretrained Neural Machine
Translation model to automatically translate the captions to Albanian. It makes use of an existing
Encoder-Decoder architecture and transfer learning to leverage pretrained image features in the Encoder
part of the architecture to both reduce computational complexity and to increase the generalizability
of the approach. The results **(BLEU-4 score of 12.6)** are close to those of works using similar architectures
for English captions and show that Albanian is a more complex language compared to English.

## Samples

<p align="center">
  <img src="/samples/best_1.png" width="250">
  <img src="/samples/best_2.png" width="250">
</p>


<p align="center">
  <img src="/samples/bad_1.png" width="250">
  <img src="/samples/bad_2.png" width="250">
</p>
