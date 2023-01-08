# Final Project - Transformer implementation

Public repository and stub/testing code for Final Project of 10-714.

To sum up what we learned during *10-714: Deep Learning Systems*, we've implemented the Transformer architecture and its corresponding modules with our self-made *needle* library for deep learning.

The overall goal of our *Final Project* is to implement the trainable Transformer architecture [1], which can be divided into some ingredients — Multi-Head Attention, Self-Attention and Positional Encoding, and The Transformer Architecture (Positionwise Feed-Forward Networks, Residual Connection and Layer Normalization, Transformer Encoder Block & Encoder, Transformer Decoder Block & Decoder, and Encoder-Decoder Seq2Seq model.)

## Project structure

- `Final project.ipynb` - notebook with report of project results
- `python/needle` - all source code for needle library and project classes
- `src` - backend c++ sources
- `tests` - tests over implemented functionalities

To explore project details, go to `Final project.ipynb`. To run it, you can follow **Setup** cell blocks to run locally or in google colab. Running locally might need some changes in makefile depending on your set up.

## Authours
* Yuxuan Sun: <yuxuan_eric_sun@outlook.com>
* Sergey: <seriy.karp2@gmail.com>
* Haitao Gao: <haitaogao423@gmail.com>