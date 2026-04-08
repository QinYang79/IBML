import torch.nn as nn
from .backbone_2D3D import DGCNN, Img_FC, SingleViewNet
from .fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
   
    
class TTClassifier(nn.Module):
    def __init__(self, args, n_classes):
        super(TTClassifier, self).__init__()

        fusion = args.fusion_method
 
        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))
 
        self.image_net = Img_FC()
        self.point_net = DGCNN(k=20, emb_dims=512)
        
    def forward(self, image, pointcloud):
        pointcloud = pointcloud.permute(0, 2, 1)
        p = self.point_net(pointcloud)
        i = self.image_net(image)

        i, p, out = self.fusion_module(i, p)

        return i, p, out


class ModelNetClassifier(nn.Module):
    def __init__(self, args, n_classes):
        super(ModelNetClassifier, self).__init__()

        fusion = args.fusion_method
 
        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))
 
        self.image_net = SingleViewNet()
        self.point_net = DGCNN(k=20, emb_dims=512)
        

        
    def forward(self, image, image2, image3, image4, pointcloud):
        pointcloud = pointcloud.permute(0, 2, 1)
        p = self.point_net(pointcloud)

        i = self.image_net(image, image2)
        i2 = self.image_net(image3, image4)
        i = 0.5 * (i + i2)

        i, p, out = self.fusion_module(i, p)

        return i, p, out