from torch.utils.data import Dataset
import h5py
import numpy as np
import random


class MNISTDataset(Dataset):
    def __init__(self, num_classes=10, partition='train'):
        self.num_classes = num_classes
    
        if partition == 'train':
            self.img = np.zeros((5000,3,30,30))
            self.pt = np.zeros((5000,1024,3))
            self.label = np.zeros(5000)
            # self.img_feat = np.load('../3d22d_noise/extracted_features/feature/3D_MNIST/train_img_feat.npy')
            with h5py.File("./data/3D_MNIST/train_point_clouds.h5", "r") as hf:    
                for i in range(5000):
                    
                    a = hf[str(i)]
                    self.img[i,:,:] = np.array(a["img"][:])
                    temp_pt = np.array(a["points"][:],dtype = 'float64')
                    self.label[i] = np.array(a.attrs["label"],dtype = 'int64')
                    ran = random.sample(range(0, temp_pt.shape[0]-1),1024)
                    self.pt[i,:,:] = temp_pt[ran,:]
            self.pt.dtype = np.float64
            
        else:
            self.img = np.zeros((1000,3,30,30))
            self.pt = np.zeros((1000,1024,3))
            self.label = np.zeros(1000)
            # self.img_feat = np.load('../3d22d_noise/extracted_features/feature/3D_MNIST/test_img_feat.npy')
            with h5py.File("./data/3D_MNIST/test_point_clouds.h5", "r") as hf:    
                for j in range(1000):
                    a = hf[str(j)]
                    self.img[j,:,:] = np.array(a["img"][:])
                    temp_pt = np.array(a["points"][:],dtype = 'float64')
                    self.label[j] = np.array(a.attrs["label"],dtype = 'int64')
                    ran = random.sample(range(0, temp_pt.shape[0]-1),1024)
                    self.pt[j,:,:] = temp_pt[ran,:]
            self.pt.dtype = np.float64

    def __getitem__(self, item):
        img_list =  self.img[item]
        pt_list =  self.pt[item]
        label = self.label[item]
        a_n, i_n = np.random.normal(0, img_list.std(), img_list.shape), np.random.normal(0,  pt_list.std(), pt_list.shape)
        return img_list, pt_list, a_n, i_n, int(label), item
    def __len__(self):
        return self.label.shape[0]