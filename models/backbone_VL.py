import numpy as np
import torch.nn as nn 
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from .resnet import ResNet152, ResNet18, ResNet34, ResNet50


class TextEncoder(nn.Module):
    def __init__(self,embed_size):
        super(TextEncoder, self).__init__()
        self.embed_size = embed_size
        self.rnn = nn.GRU(300, self.embed_size, 1, batch_first=True, bidirectional=True)
 
    def forward(self, x_emb, lengths):
        """Handles variable size captions
        """ 
        lengths = lengths.cpu()
        # Embed word ids to vectors
        self.rnn.flatten_parameters()
        packed = pack_padded_sequence(x_emb, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded
        cap_emb = (cap_emb[:, :, :cap_emb.size(2) // 2] + cap_emb[:, :, cap_emb.size(2) // 2:]) / 2

        # I = torch.LongTensor(lengths).view(-1, 1, 1)
        # I = Variable(I.expand(x_emb.size(0), 1, self.embed_size)-1).cuda()
        # out = torch.gather(padded[0], 1, I).squeeze(1)

        # return out
 
        out = maxk_pool1d_var(cap_emb, 1, cap_emb.size(1), cap_len) # avg-pool

        return out
    

def maxk(x, dim, k):
    index = x.topk(k, dim=dim)[1]
    return x.gather(dim, index)


def maxk_pool1d_var(x, dim, k, lengths):
    results = list()
    lengths = list(lengths.cpu().numpy())
    lengths = [int(x) for x in lengths]
    for idx, length in enumerate(lengths):
        # print(k,length)
        k = min(k, length)
        max_k_i = maxk(x[idx, :length, :], dim - 1, k).mean(dim - 1)
        results.append(max_k_i)
    results = torch.stack(results, dim=0)
    return results

class ImageEncoder(nn.Module):
    def __init__(self, embed_size,type=18):
        super(ImageEncoder, self).__init__()
        self.embed_size = embed_size
        if type==18:
            self.model = ResNet18()
            
        if type==34:
            self.model = ResNet34()
            
        if type==50:
            self.model = ResNet50()
            
        if type==152:
            self.model = ResNet152()
       
    def forward(self, x):
        return self.model(x)
    

 