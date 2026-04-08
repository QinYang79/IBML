import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset
import nltk
 
class VLlDataset(Dataset):
    def __init__(self, args, glove_embeddings, mode='train', labels_map=None):
        self.args = args
        self.image = []
        self.text = []
        self.label = []
        self.mode = mode
        self.glove_embeddings = glove_embeddings
        self.max_seq_len = 77

        if 'MVSA' in args.dataset:

            self.data_root = args.data_path
            class_dict = {'negative':0, 'positive':1, 'neutral':2}

            self.train_json = os.path.join(self.data_root, args.dataset + '/train.jsonl')
            self.test_json = os.path.join(self.data_root, args.dataset + '/test.jsonl')
            self.val_json = os.path.join(self.data_root, args.dataset + '/dev.jsonl')
        
            if mode == 'train':
                json_file = self.train_json
            elif mode == 'test':
                json_file = self.test_json
            else:
                json_file = self.val_json

            self.data = [json.loads(l) for l in open(json_file)]

            self.num_classes = 3
            if labels_map is not None:
                class_dict = labels_map
                self.num_classes = len(class_dict.keys())

            file_root_path =  os.path.join(self.data_root, args.dataset)
            for line in self.data:
                img_path = os.path.join(file_root_path, line['img'])
                
                if os.path.exists(img_path): 
                    self.image.append(img_path) 
                    self.text.append(line['text'])
                    self.label.append(class_dict[line['label']])
                else:
                    continue

        if 'food101'  in args.dataset:
            self.data_root = args.data_path

            self.train_json = os.path.join(self.data_root, args.dataset + '/train.jsonl')
            self.test_json = os.path.join(self.data_root, args.dataset + '/test.jsonl')
            self.val_json = os.path.join(self.data_root, args.dataset + '/dev.jsonl')
        

            if mode == 'train':
                json_file = self.train_json
            elif mode == 'test':
                json_file = self.test_json
            else:
                json_file = self.val_json

            self.data = [json.loads(l) for l in open(json_file)]
            
 
            self.num_classes = 101

            if labels_map is not None:
                class_dict = labels_map
                self.num_classes = len(class_dict.keys())

            file_root_path =  os.path.join(self.data_root, args.dataset)
            for line in self.data:
                img_path = os.path.join(file_root_path, line['img'])
                
                if os.path.exists(img_path): 
                    self.image.append(img_path) 
                    self.text.append(line['text'])
                    self.label.append(class_dict[line['label']])
                else:
                    continue
        if 'wiki' in args.dataset:
            self.data_root = args.data_path        
            json_file = os.path.join(self.data_root, args.dataset + '/train.jsonl')        
            self.data = [json.loads(l) for l in open(json_file)]
            
            if labels_map is not None:
                class_dict = labels_map
                self.num_classes = len(class_dict.keys())

            file_root_path =  os.path.join(self.data_root, args.dataset)
            for line in self.data:
                img_path = os.path.join(file_root_path, 'wikipedia_dataset/images', line['img'])
                if os.path.exists(img_path) and line['mode']==mode: 
                    self.image.append(img_path) 
                    self.text.append(line['text'])
                    self.label.append(class_dict[line['label']])
                else:
                    continue
                
            self.num_classes = 10
                
        print(f'{mode} samples==============>{len(self.image)}')
     
        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
  
    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):

        # image
        # idx1 = self.noisy_idx[idx]
        img = Image.open(self.image[idx]).convert('RGB')
        img = self.transform(img)
 
        # text 
        caption = self.text[idx]
        tokens = nltk.tokenize.word_tokenize( str(caption).lower())
 
        sentence = torch.LongTensor(
            [
                self.glove_embeddings.vocab.stoi[w] if w in self.glove_embeddings.vocab.stoi else self.glove_embeddings.vocab.stoi["[UNK]"] for w in tokens
            ]
        ) 

        # glove embedding
        g_sentence = self.glove_embeddings.forward(sentence).cpu()

        # label
        label = self.label[idx]

        # noise
        img_ns = torch.from_numpy(np.random.normal(0,  img.std(), img.shape))
        txt_ns = torch.from_numpy(np.random.normal(0, g_sentence.std(), g_sentence.shape))
 
        return g_sentence, img, txt_ns, img_ns, label, idx
    
def collate_fn(data): 
    data.sort(key=lambda x: x[0].size(0), reverse=True) 
    txts, imgs, txt_ns, img_ns, labels, idxs = zip(*data)
    imgs = torch.stack(imgs, 0).float() 
    img_ns = torch.stack(img_ns, 0).float() 
 
    labels = torch.Tensor(labels).long()
    idxs = torch.Tensor(idxs).long()
 
    lengths = [cap.size(0) for cap in txts]
    targets = torch.zeros(len(txts), max(lengths), txts[0].size(1)).float()
    target_ns = torch.zeros(len(txts), max(lengths), txt_ns[0].size(1)).float()

    for i, cap in enumerate(txts):
        end = lengths[i]
        if end==0:
            lengths[i] = 1 
        else:
            targets[i, :end] = cap[:end]
            target_ns[i, :end] = txt_ns[i][:end]

    lengths = torch.Tensor(lengths).long() 
    return targets, imgs, target_ns, img_ns, labels, idxs, lengths
