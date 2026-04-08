import argparse
import json
import logging
import os 

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 

from dataset.VLdataset import VLlDataset, collate_fn 
from models.basic_model_VL import VLClassifier
from utils.utils import setup_seed, weight_init
from utils.glove_encoder import GloveBowEncoder

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def tensor_cosine(x,y):
    return l2norm(x) @ l2norm(y).t()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MVSA_Single', type=str,
                        help='MVSA_Single, food101, wiki')
    
    parser.add_argument('--add_noise', default=0, type=int) 
    parser.add_argument('--warmup', default=0, type=int)
    parser.add_argument('--epsilon', default=0, type=int)

    parser.add_argument('--modulation', default='OGM_GE', type=str,

                        choices=['Normal', 'OGM', 'OGM_GE'])
    parser.add_argument('--fusion_method', default='NC', type=str,
                        choices=['sum', 'concat', 'gated', 'film', 'NC'])
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--use_video_frames', default=3, type=int)
    parser.add_argument('--data_path', default='', type=str) 

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--modulation_starts', default=0, type=int, help='where modulation begins')
    parser.add_argument('--modulation_ends', default=50, type=int, help='where modulation ends')
    parser.add_argument('--alpha', required=True, type=float, help='alpha in OGM-GE')

    parser.add_argument('--ckpt_path', required=True, type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--use_tensorboard', default=False, type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', type=str, help='path to save tensorboard logs')

    parser.add_argument('--gpu_ids', default='0', type=str, help='GPU ids')

    return parser.parse_args()

 
soft_labels = torch.tensor([]).cuda()
iter_ = 0
discrepancy_ratio = []
r=1
stds = []
last_raitos = [1,1]


def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler, writer=None,logger=None, scale=10):
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    
    model.train()
    logger.info("Start training ... ")

    _loss = 0
    _loss_t = 0
    _loss_v = 0
    _loss_f = 0
 
    global soft_labels, iter_, discrepancy_ratio, r, last_raitos, stds
    
    class_number = dataloader.dataset.num_classes

    if soft_labels.size(0) == 0:
        soft_labels = torch.zeros((3,dataloader.dataset.__len__(), class_number))
        stds =  torch.zeros((2,dataloader.dataset.__len__()))
    
    for step, (cap, img, cap_n, img_n, label, ids, length) in enumerate(dataloader):
        iter_ += 1
        if iter_%40==0:
            logger.info(f'=============>r: {r}')
 
        cap, img, label, length = cap.to(device), img.to(device), label.to(device), length.to(device)
 
      
        y = torch.nn.functional.one_hot(label, class_number).float().to(device)
        if epoch == 0:
            y1 = y
            y2 = y
        else: 
            y1 = soft_labels[0][ids].to(device)
            y2 = soft_labels[1][ids].to(device)
        bs = cap.size(0) 
        pred1 = (y1.argmax(dim=1).to(device) == label).float()
        pred2 = (y2.argmax(dim=1).to(device) == label).float()

        score_t =  torch.tensor([y1[i][label[i]] for i in range(bs)]).to(device).sum().data.item() 
        score_v =  torch.tensor([y2[i][label[i]] for i in range(bs)]).to(device).sum().data.item() 
       
        e = 10
        at =  2 * score_t / (score_t + score_v)
        av =  2 * score_v / (score_t + score_v)   
        r = score_t/score_v
 
        if args.modulation_starts < epoch < args.modulation_ends and args.add_noise == 1: 
            t_noise = torch.normal(0, cap[:,:min(length),:].std(), size= cap[0].size()).cuda().unsqueeze(0).repeat(bs,1,1) 
            v_noise = torch.normal(0, img.std(), size= img[0].size()).cuda().unsqueeze(0).repeat(bs,1,1,1) 

            if at > 1: 
                p =  np.clip(at, 0, e)/e
                mask =  ((torch.rand(bs).to(device) < p).float() * pred1).unsqueeze(-1).unsqueeze(-1).expand_as(cap) 
                cap = cap + t_noise * mask

            if av > 1:  
                p =  np.clip(av, 0, e)/e  
                mask =  ((torch.rand(bs).to(device) < p).float() * pred2).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(img)  
                img = img + v_noise * mask

        optimizer.zero_grad()
        t, v, out = model.module.forward(cap, img, length)
     
        if args.fusion_method in ['sum']:
            out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_y.weight, 0, 1)) +
                    model.module.fusion_module.fc_y.bias)
            out_t = (torch.mm(t, torch.transpose(model.module.fusion_module.fc_x.weight, 0, 1)) +
                    model.module.fusion_module.fc_x.bias)
        else:
            weight_size = model.module.fusion_module.fc_out.weight.size(1) 
            out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1))
                    + model.module.fusion_module.fc_out.bias / 2)

            out_t = (torch.mm(t, torch.transpose(model.module.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1))
                    + model.module.fusion_module.fc_out.bias / 2)
 
        loss_f = criterion(out, label)
        loss_v = criterion(out_v, label)
        loss_t = criterion(out_t, label)
        
        p1 = softmax(out_t)
        p2 = softmax(out_v)  
        p = softmax(out)  
 
        loss = loss_f 
        if args.modulation_starts < epoch < args.modulation_ends:
            if at < 1: 
                loss =  loss_t + loss_f
            if av < 1: 
                loss =  loss_v + loss_f
            if (av + at) == 0:
                loss = loss_f   
        else:
            loss = loss_f   
        loss.backward() 

        soft_labels[0][ids] = p1.detach().cpu()
        soft_labels[1][ids] = p2.detach().cpu() 
        soft_labels[2][ids] = p.detach().cpu() 
 
        score_t = sum([p1[i][label[i]].item() for i in range(bs)])
        score_v = sum([p2[i][label[i]].item() for i in range(bs)])
        last_raitos =  [score_t, score_v]
        discrepancy_ratio.append(score_t/score_v)

        optimizer.step()

        _loss += loss.item()
        _loss_t += loss_t.item()
        _loss_v += loss_v.item()
        _loss_f += loss_f.item()
    
    if args.warmup:
        if epoch < 35:
            lr = min(args.learning_rate *  (epoch / 2 + 1), args.learning_rate * 10) # 5-th
        elif epoch < args.lr_decay_step: 
            lr = args.learning_rate
        else:
            lr = args.learning_rate * args.lr_decay_ratio
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr 
    else:
        scheduler.step()

    logger.info(f'=============>lr: {optimizer.param_groups[0]["lr"]}')
 
    return _loss / len(dataloader), _loss_t / len(dataloader), _loss_v / len(dataloader)


def valid(args, model, device, dataloader):

    softmax = nn.Softmax(dim=1)
    n_classes =  dataloader.dataset.num_classes
    with torch.no_grad():
        model.eval()
        # TODO: more flexible
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_t = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)] 
        for step, (cap, img, cap_n, img_n, label, idx, length) in enumerate(dataloader):

            cap, img, label, idx, length = cap.to(device), img.to(device), label.to(device), idx.to(device), length.to(device)

            t, v, out = model.module.forward(cap, img, length)
        
            if args.fusion_method == 'sum':
                out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_y.weight, 0, 1)) +
                        model.module.fusion_module.fc_y.bias )
                out_t = (torch.mm(t, torch.transpose(model.module.fusion_module.fc_x.weight, 0, 1)) +
                        model.module.fusion_module.fc_x.bias )
            else:
                weight_size = model.module.fusion_module.fc_out.weight.size(1)
                out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1)) +
                        model.module.fusion_module.fc_out.bias / 2)
                out_t = (torch.mm(t, torch.transpose(model.module.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1)) +
                        model.module.fusion_module.fc_out.bias / 2)
            
            prediction = softmax(out)
            pred_v = softmax(out_v)
            pred_t= softmax(out_t) 

            for i in range(img.shape[0]):

                ma = np.argmax(prediction[i].cpu().data.numpy())
                v = np.argmax(pred_v[i].cpu().data.numpy())
                a = np.argmax(pred_t[i].cpu().data.numpy())
                num[label[i]] += 1.0
 
                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0 
                if np.asarray(label[i].cpu()) == v:
                    acc_v[label[i]] += 1.0  
                if np.asarray(label[i].cpu()) == a:
                    acc_t[label[i]] += 1.0 
    

    return sum(acc) / sum(num), sum(acc_t) / sum(num), sum(acc_v) / sum(num)

def save_config(opt, file_path):
    with open(file_path, "w") as f:
        json.dump(opt.__dict__, f, indent=2)

def init_logging(log_file_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s %(message)s')

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def main():
    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')

    model = VLClassifier(args)

    model.apply(weight_init)
    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    model.cuda() 

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    data_root = '/home/qinyang/projects/OGM_GE/data/'
    train_json = os.path.join(data_root, args.dataset + '/train.jsonl') 
    data = [json.loads(l) for l in open(train_json)]
    
    labels = []
    for line in data:
        label = line['label']
        if label not in labels:
           labels.append(label)
 
    labels_map = dict()
    for i,v in enumerate(labels):
        labels_map[v] = i 
 
    glove_embeddings = GloveBowEncoder() 
    train_dataset = VLlDataset(args, glove_embeddings, mode='train',labels_map=labels_map)
    test_dataset = VLlDataset(args, glove_embeddings, mode='test',labels_map=labels_map)
    val_dataset = VLlDataset(args, glove_embeddings, mode='val',labels_map=labels_map) 
     

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,collate_fn=collate_fn,
                                  shuffle=True, num_workers=16, pin_memory=True)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn,
                                 shuffle=False, num_workers=16, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,collate_fn=collate_fn,
                                 shuffle=False, num_workers=16, pin_memory=True)
    
    if args.train:
        if not os.path.isdir(args.ckpt_path):
            os.makedirs(args.ckpt_path)
        save_config(args, os.path.join(args.ckpt_path, "config.json"))
    # logger initialization 
    logger = init_logging(args.ckpt_path + f'/log_{args.train}.txt')
    logger.info(f"===>PID:{os.getpid()}, GPU:[{args.gpu_ids}]")
    logger.info(args)
    # Load Vocabulary

    if args.train:

        best_acc = 0.0

        for epoch in range(args.epochs):

            logger.info('Epoch: {}: '.format(epoch))

            if args.use_tensorboard:

                writer_path = os.path.join(args.tensorboard_path, args.dataset)
                if not os.path.exists(writer_path):
                    os.mkdir(writer_path)
                log_name = '{}_{}'.format(args.fusion_method, args.modulation)
                writer = SummaryWriter(os.path.join(writer_path, log_name))

                batch_loss, batch_loss_t, batch_loss_v = train_epoch(args, epoch, model, device,
                                                                     train_dataloader, optimizer, scheduler)
                acc, acc_t, acc_v = valid(args, model, device, val_dataloader)

                writer.add_scalars('Loss', {'Total Loss': batch_loss,
                                            'Audio Loss': batch_loss_t,
                                            'Visual Loss': batch_loss_v}, epoch)

                writer.add_scalars('Evaluation', {'Total Accuracy': acc,
                                                  'Text Accuracy': acc_t,
                                                  'Visual Accuracy': acc_v}, epoch)

            else:
                batch_loss, batch_loss_t, batch_loss_v = train_epoch(args, epoch, model, device,
                                                                     train_dataloader, optimizer, scheduler,logger=logger)
                acc, acc_t, acc_v = valid(args, model, device, val_dataloader)

            if acc > best_acc:
                best_acc = float(acc)

                if not os.path.exists(args.ckpt_path):
                    os.mkdir(args.ckpt_path)

                model_name = f'model_bset_VL.pth'

                saved_dict = {'saved_epoch': epoch,
                              'modulation': args.modulation,
                              'alpha': args.alpha,
                              'fusion': args.fusion_method,
                              'acc': acc,
                              'model': model.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict()}

                save_dir = os.path.join(args.ckpt_path, model_name)

                torch.save(saved_dict, save_dir)
                logger.info('The best model has been saved at {}.'.format(save_dir))
                logger.info("Loss: {:.3f}, Acc: {:.3f}".format(batch_loss, acc))
                logger.info("Text Acc: {:.3f}, Image Acc: {:.3f} Fusion Acc: {:.3f}".format(acc_t, acc_v, acc))
            else:
                logger.info("Loss: {:.3f}, Acc: {:.3f}, Best Acc: {:.3f}".format(batch_loss, acc, best_acc))
                logger.info("Text Acc: {:.3f}, Image Acc: {:.3f} Fusion Acc: {:.3f}".format(acc_t, acc_v, acc))
    
    else:
        # first load trained model
        loaded_dict = torch.load(args.ckpt_path+f'/model_bset_VL.pth')
        # epoch = loaded_dict['saved_epoch']
        modulation = loaded_dict['modulation']
        # alpha = loaded_dict['alpha']
        fusion = loaded_dict['fusion']
        state_dict = loaded_dict['model']
        # optimizer_dict = loaded_dict['optimizer']
        # scheduler = loaded_dict['scheduler']

        assert modulation == args.modulation, 'inconsistency between modulation method of loaded model and args !'
        assert fusion == args.fusion_method, 'inconsistency between fusion method of loaded model and args !'

        model.load_state_dict(state_dict)
        logger.info('Trained model loaded!')

        acc, acc_t, acc_v = valid(args, model, device, test_dataloader)
        logger.info('Accuracy: {}, accuracy_t: {}, accuracy_v: {}'.format(acc, acc_t, acc_v))


if __name__ == "__main__":
    main()
