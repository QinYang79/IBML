import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone_VL import ImageEncoder,TextEncoder
from .fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion


class VLClassifier(nn.Module):
    def __init__(self, args):
        super(VLClassifier, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'food101':
            n_classes = 101
        elif args.dataset == 'MVSA_Single':
            n_classes = 3
        elif args.dataset == 'wiki':
            n_classes = 10
 
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))


        type_ = 18
        es = 512
        if type_ > 34:
            es = 2048

        if fusion == 'sum':
            self.fusion_module = SumFusion(input_dim = es,output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(input_dim = es*2,output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(input_dim = es,output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(input_dim = es,output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        self.image_net = ImageEncoder(es,type_)
        self.text_net = TextEncoder(es)
 
    def forward(self, text, image, text_length): 
        t = self.text_net(text,text_length)
        v = self.image_net(image)
   
        t, v, out = self.fusion_module(t, v)
        return t, v, out

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

