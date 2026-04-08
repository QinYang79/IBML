import copy
import csv
import os
import pickle
import random
import librosa
import numpy as np
from scipy import signal
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pdb

class AVDataset(Dataset):

    def __init__(self, args, mode='train'):
        self.args = args
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode

        self.data_root = args.data_path
        self.dataset =  args.dataset
 
        if args.dataset == 'AVE':
            self.train_txt = os.path.join(self.data_root, args.dataset + '/trainSet.txt')
            self.test_txt = os.path.join(self.data_root, args.dataset + '/testSet.txt')
            self.val_txt = os.path.join(self.data_root, args.dataset + '/valSet.txt')
            
            labels = []
            with open(self.test_txt, 'r') as f1:
                    files = f1.readlines()
                    for item in files:
                        item = item.split('&')
                        if item[0] not in labels:
                            labels.append(item[0])
            class_dict = dict()
            for i,v in enumerate(labels):
                class_dict[v] = i  
            self.class_number = len(labels)        
        
            if mode == 'train':
                txt_file = self.train_txt
            elif mode == 'test':
                txt_file = self.test_txt
            else:
                txt_file = self.val_txt
    
            with open(txt_file, 'r') as f2:
                files = f2.readlines()
                for item in files:
                    item = item.split('&')
                    audio_path = os.path.join('./data/AVE/Audio-1004', item[1] + '.pkl')
                    visual_path = os.path.join('./data/AVE', 'Image-{:02d}-FPS'.format(self.args.fps), item[1])

                    if os.path.exists(audio_path) and os.path.exists(visual_path):
                        if audio_path not in self.audio:
                            self.image.append(visual_path)
                            self.audio.append(audio_path)
                            self.label.append(class_dict[item[0]])
                    else:
                        continue
       
        elif args.dataset == 'CREMAD':
            self.train_csv = os.path.join(self.data_root, args.dataset + '/train.csv')
            self.test_csv = os.path.join(self.data_root, args.dataset + '/test.csv')
 
            labels = []
            with open(self.test_csv, encoding='UTF-8-sig') as f2:
                csv_reader = csv.reader(f2)  
                for item in csv_reader:
                    if item[1] not in labels:
                        labels.append(item[1])

            class_dict = dict()
            for i,v in enumerate(labels):
                class_dict[v] = i  
            self.class_number = len(labels)            

            if mode == 'train':
                csv_file = self.train_csv
            else:
                csv_file = self.test_csv
    
            with open(csv_file, encoding='UTF-8-sig') as f2:
                csv_reader = csv.reader(f2)  
                for item in csv_reader:
                    audio_path = os.path.join('./data/CREMAD/CREMA-D/AudioWAV', item[0] + '.wav')

                    visual_path = os.path.join('./data/CREMAD/CREMA-D', 'Image-{:02d}-FPS'.format(self.args.fps), item[0])
    
                    if os.path.exists(audio_path) and os.path.exists(visual_path): 
    
                        self.image.append(visual_path)
                        self.audio.append(audio_path)
                        self.label.append(class_dict[item[1]])
                    else:
                        continue

        elif args.dataset == 'avsbench':

            self.full_csv = os.path.join(self.data_root, args.dataset,  's4_meta_data.csv')
            labels = []
            with open(self.full_csv, 'r') as f:
                files = f.readlines()
            for item in files[1:]:
                item = item.split(',')
                item[3] = item[3].replace('\n','')
                if item[2] not in labels:
                    labels.append(item[2])
            class_dict = dict()
            for i,v in enumerate(labels):
                class_dict[v] = i  
            self.class_number = len(labels)   

 
            for class_ in labels: 
                base_path = os.path.join(self.data_root, args.dataset, f's4_data/audio_wav/{mode}/{class_}')   
                av_files = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]
                for file in av_files: 
                    inst_name = file.replace('.wav','') 
                    audio_path = os.path.join(self.data_root, args.dataset, f's4_data/audio_wav/{mode}/{class_}/{inst_name}.wav')  
                    visual_path = os.path.join(self.data_root, args.dataset, f's4_data/visual_frames/{mode}/{class_}/{inst_name}')  
                    if os.path.exists(audio_path) and os.path.exists(visual_path): 
                        self.image.append(visual_path)
                        self.audio.append(audio_path)
                        self.label.append(class_dict[class_])
                    else: 
                        continue

        elif args.dataset == 'VGGSound50':

            self.full_csv = os.path.join(self.data_root, args.dataset,  'train.txt')
            labels = []
            with open(self.full_csv, 'r') as f:
                files = f.readlines()

            for item in files:
                item = item.split(',') 
                if item[1] not in labels:
                    labels.append(item[1])
            class_dict = dict()
            for i,v in enumerate(labels):
                class_dict[v] = i  
            self.class_number = len(labels)   
      
            with open(os.path.join(self.data_root, args.dataset,  f'{mode}.txt'), 'r') as f:
                files = f.readlines()

                for item in files:
                    item = item.split(',') 
                    if item[1] in labels:
                        audio_path = os.path.join(self.data_root, args.dataset,  'audios', item[0].replace(".mp4",'.wav')) 
                        visual_path =  os.path.join(self.data_root, args.dataset,  'Image-01-FPS', item[0].replace(".mp4",'')) 
                        # print(audio_path,visual_path)
                        if os.path.exists(audio_path) and os.path.exists(visual_path): 
                            self.image.append(visual_path)
                            self.audio.append(audio_path)
                            self.label.append(class_dict[item[1]])
                        else:
                            continue
    
        elif args.dataset == 'Kinetics-Sounds':
            self.full_csv = os.path.join(self.data_root, args.dataset,  'train.txt')
            labels = []
            with open(self.full_csv, 'r') as f:
                files = f.readlines()

            for item in files:
                item = item.split(',') 
                item[2] = item[2].replace('\n','')
                if item[2] not in labels:
                    labels.append(item[2])
            class_dict = dict()
            for i,v in enumerate(labels):
                class_dict[v] = i  
            self.class_number = len(labels)   
 
            with open(os.path.join(self.data_root, args.dataset,  f'{mode}.txt'), 'r') as f:
                files = f.readlines()
                for item in files:
                    item = item.split(',') 
                    item[2] = item[2].replace('\n','')
                    if item[2] in labels:
                        audio_path = os.path.join(self.data_root, args.dataset,  mode, 'audios', item[0]) 
                        visual_path =  os.path.join(self.data_root, args.dataset, mode, 'Image-01-FPS', item[1]) 
 
                        if os.path.exists(audio_path) and os.path.exists(visual_path): 
                            self.image.append(visual_path)
                            self.audio.append(audio_path)
                            self.label.append(class_dict[item[2]])
                        else:
                            continue
    

        elif args.dataset == 'UCF101':
            pp =  os.path.join(self.data_root, args.dataset, 'trainlist01.txt')   
            labels = []
            with open(pp, 'r') as f:
                files = f.readlines()
            for item in files:
                item = item.split('/')[0]
                if item not in labels:
                    labels.append(item)
            class_dict = dict()
            for i,v in enumerate(labels):
                class_dict[v] = i  
            self.class_number = len(labels)       
            pp1 =  os.path.join(self.data_root, args.dataset, f'{mode}list01.txt')  
            pp2 =  os.path.join(self.data_root, args.dataset, f'{mode}list02.txt')  
            pp3 =  os.path.join(self.data_root, args.dataset, f'{mode}list03.txt')  


            with open(pp1, 'r') as f:
                files1 = f.readlines()
            with open(pp2, 'r') as f:
                files2 = f.readlines()
            with open(pp3, 'r') as f:
                files3 = f.readlines()
            files = files1 + files2 + files3

            for item in files:
                item = item.split('/')
                label = item[0] 
                item = item[1].replace('\n','')
                item = item.split(' ')[0].replace('.avi','')
                
                audio_path = os.path.join('./data/UCF101/audio_wav', item + '.wav')
                visual_path = os.path.join('./data/UCF101/visual_frames', f'{item}')

                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.image.append(visual_path)
                    self.audio.append(audio_path)
                    self.label.append(class_dict[label])
                else:
                    continue

       


        print(f'{mode} samples==============>{len(self.image)} classes {self.class_number}')
      

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):

        # audio 
        if self.dataset == 'AVE':
            spectrogram = pickle.load(open(self.audio[idx], 'rb')) 
 
        elif self.dataset == 'Kinetics-Sounds':
            # audio
            sample, rate = librosa.load(self.audio[idx], sr=16000, mono=True)
            while len(sample)/rate < 10.:
                sample = np.tile(sample, 2)

            start_point = random.randint(a=0, b=rate*5)
            new_sample = sample[start_point:start_point+rate*5]
            new_sample[new_sample > 1.] = 1.
            new_sample[new_sample < -1.] = -1.

            spectrogram = librosa.stft(new_sample, n_fft=256, hop_length=128)
            spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        else:
            samples, rate = librosa.load(self.audio[idx], sr=22050)
            resamples = np.tile(samples, 3)[:22050*3]
            resamples[resamples > 1.] = 1.
            resamples[resamples < -1.] = -1.

            spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
            spectrogram = np.log(np.abs(spectrogram) + 1e-7)  
        
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual 
        # image_samples = os.listdir(self.image[idx])
        # fps = len(image_samples)
        # if fps > self.args.use_video_frames:
        #     fps = self.args.use_video_frames
 
        # select_index = np.random.choice(len(image_samples), size= fps, replace=False)
        # select_index.sort()
        # images = torch.zeros((fps, 3, 224, 224))
        # for i, v in enumerate(select_index):
        #     img = Image.open(os.path.join(self.image[idx], image_samples[v])).convert('RGB')
        #     img = transform(img)
        #     images[i] = img

        # images = torch.permute(images, (1,0,2,3)) 

        image_samples = os.listdir(self.image[idx])
        file_num = len(image_samples)
        pick_num = 3
        seg = file_num//pick_num 
        images = torch.zeros((pick_num, 3, 224, 224))
        for i in range(pick_num):
            if self.mode == 'train':
                index = random.randint(i*seg + 1, i*seg + seg)
            else:
                index = i*seg + seg//2
            img = Image.open(os.path.join(self.image[idx], image_samples[index-1])).convert('RGB')
            images[i] = transform(img) 
        images = torch.permute(images, (1,0,2,3))

        label = self.label[idx]
        return spectrogram, images, np.random.normal(0, spectrogram.std(), spectrogram.shape), np.random.normal(0,  images.std(), images.shape), label, idx
    
        
 