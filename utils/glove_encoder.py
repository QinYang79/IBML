#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import numpy as np
import torchtext

class Vocab(object):
    def __init__(self, emptyInit=False):
        if emptyInit:
            self.stoi, self.itos, self.vocab_sz = {}, [], 0
        else:
            self.stoi = {
                w: i
                for i, w in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
            }
            self.itos = [w for w in self.stoi]
            self.vocab_sz = len(self.itos)

    def add(self, words):
        cnt = len(self.itos)
        for w in words:
            if w in self.stoi:
                continue
            self.stoi[w] = cnt
            self.itos.append(w)
            cnt += 1
        self.vocab_sz = len(self.itos)


 
class GloveBowEncoder(nn.Module):
    def __init__(self, glove_path='/home/qinyang/projects/TCL/vocab/vector_cache/'):
        super(GloveBowEncoder, self).__init__() 
        self.glove_path= glove_path
        glove = torchtext.vocab.GloVe(cache=glove_path)

        self.vocab = self.get_vocab(glove)
        self.vocab_sz = self.vocab.vocab_sz
        self.embed_sz = 300

        self.embed = nn.Embedding(self.vocab_sz, self.embed_sz)

        self.load_glove(glove) 
        self.embed.weight.requires_grad = False


    def get_vocab(self, glove):
        vocab = Vocab() 
        word_list = list(glove.stoi.keys())
        vocab.add(word_list)

        return vocab

    def load_glove(self, glove):
        print("Loading glove")
        pretrained_embeds = np.zeros(
            (self.vocab_sz, self.embed_sz), dtype=np.float32
        )
     
        for i in range(self.vocab_sz):
            word = self.vocab.itos[i]
            if word in glove.stoi:
               pretrained_embeds[i] = glove.vectors[glove.stoi[word]]
             
        self.embed.weight = torch.nn.Parameter(torch.from_numpy(pretrained_embeds))
        print("Loading glove finish")

    def forward(self, x):
        return self.embed(x) 