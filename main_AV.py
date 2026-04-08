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
 
import torch.nn.functional as F
 
from dataset.AVdataset import AVDataset 
from models.basic_model_AV import AVClassifier
from utils.utils import setup_seed, weight_init

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
    parser.add_argument('--dataset', default='AVE', type=str,
                        help='CREMAD, AVE, UCF101')
    parser.add_argument('--modulation', default='OGM_GE', type=str,

                        choices=['Normal', 'OGM', 'OGM_GE'])
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film']) 

    parser.add_argument('--add_noise', default=0, type=int) 
    parser.add_argument('--warmup', default=0, type=int)
    parser.add_argument('--epsilon', default=0, type=int)
 
    parser.add_argument('--fps', default=1, type=int)  
    parser.add_argument('--use_video_frames', default=3, type=int)
    parser.add_argument('--data_path', default='/home/qinyang/projects/OGM_GE/data', type=str) 
    
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

    parser.add_argument('--gpu_ids', default='0, 1', type=str, help='GPU ids')



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
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()

    model.train()
    logger.info("Start training ... ")

    _loss = 0
    _loss_a = 0
    _loss_v = 0
    _loss_f = 0
 
    global soft_labels, iter_, discrepancy_ratio, r, last_raitos, stds
    
    class_number = dataloader.dataset.class_number

    if soft_labels.size(0) == 0:
        soft_labels = torch.zeros((3,dataloader.dataset.__len__(), class_number))
        stds =  torch.zeros((2,dataloader.dataset.__len__()))
    
    for step, (spec, image, anoise, vnoise, label, ids) in enumerate(dataloader):
        iter_ += 1
        if iter_%40==0:
            logger.info(f'=============>r: {r}')
 
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)  

        y = torch.nn.functional.one_hot(label, class_number).float().to(device)
        if epoch == 0:
            y1 = y
            y2 = y
        else: 
            y1 = soft_labels[0][ids].to(device)
            y2 = soft_labels[1][ids].to(device)
        bs = spec.size(0) 
        pred1 = (y1.argmax(dim=1).to(device) == label).float()
        pred2 = (y2.argmax(dim=1).to(device) == label).float()

        score_a =  torch.tensor([y1[i][label[i]] for i in range(bs)]).to(device).sum().data.item() 
        score_v =  torch.tensor([y2[i][label[i]] for i in range(bs)]).to(device).sum().data.item() 
       
        e = 10 
        aa =  2 * score_a / (score_a + score_v)
        av =  2 * score_v / (score_a + score_v)   
        r = score_a/score_v
 
        if args.modulation_starts < epoch < args.modulation_ends and args.add_noise == 1: 
            a_noise = torch.normal(0, spec.std(), size= spec[0].size()).cuda().unsqueeze(0).repeat(bs,1,1) 
            v_noise = torch.normal(0, image.std(), size= image[0].size()).cuda().unsqueeze(0).repeat(bs,1,1,1,1) 

            if aa > 1: 
                p =  np.clip(aa, 0, e)/e
                mask =  ((torch.rand(bs).to(device) < p).float() * pred1).unsqueeze(-1).unsqueeze(-1).expand_as(spec) 
                spec = spec + a_noise * mask

            if av > 1:  
                p =  np.clip(av, 0, e)/e  
                mask =  ((torch.rand(bs).to(device) < p).float() * pred2).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(image)  
                image = image + v_noise * mask

        optimizer.zero_grad()
        a, v, out  = model.module.forward(spec.unsqueeze(1).float(), image.float())

        if args.fusion_method in ['sum']:
            out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_y.weight, 0, 1)) +
                    model.module.fusion_module.fc_y.bias)
            out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_x.weight, 0, 1)) +
                    model.module.fusion_module.fc_x.bias)
        else:
            weight_size = model.module.fusion_module.fc_out.weight.size(1) 
            out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1))
                    + model.module.fusion_module.fc_out.bias / 2)

            out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1))
                    + model.module.fusion_module.fc_out.bias / 2)
   
        loss_f = criterion(out, label)
        loss_v = criterion(out_v, label)
        loss_a = criterion(out_a, label)

        p1 = softmax(out_a)
        p2 = softmax(out_v)
        p = softmax(out)
 
        loss = loss_f 
        if args.modulation_starts < epoch < args.modulation_ends:
            if aa < 1: 
                loss =  loss_a + loss_f
            if av < 1: 
                loss =  loss_v + loss_f
            if (av + aa) == 0:
                loss = loss_f   
        else:
            loss = loss_f   
        loss.backward() 

        soft_labels[0][ids] = p1.detach().cpu()
        soft_labels[1][ids] = p2.detach().cpu() 
        soft_labels[2][ids] = p.detach().cpu() 
 
        score_a = sum([p1[i][label[i]].item() for i in range(bs)])
        score_v = sum([p2[i][label[i]].item() for i in range(bs)])
        last_raitos =  [score_a, score_v]
        discrepancy_ratio.append(score_a/score_v)

        optimizer.step()

        _loss += loss.item()
        _loss_a += loss_a.item()
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
 
    return _loss / len(dataloader), _loss_f/len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader)


def valid(args, model, device, dataloader):
    softmax = nn.Softmax(dim=1)
    n_classes = dataloader.dataset.class_number

    with torch.no_grad():
        model.eval()
        # TODO: more flexible
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

 
        for step, (spec, image, anoise, vnoise, label, ids) in enumerate(dataloader):

            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            a, v, out = model(spec.unsqueeze(1).float(), image.float(), True )
        
            if args.fusion_method == 'sum':
                out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_y.weight, 0, 1)) +
                        model.module.fusion_module.fc_y.bias )
                out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_x.weight, 0, 1)) +
                        model.module.fusion_module.fc_x.bias )
            else:
                out_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_out.weight[:, 512:], 0, 1)) +
                        model.module.fusion_module.fc_out.bias / 2)
                out_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_out.weight[:, :512], 0, 1)) +
                        model.module.fusion_module.fc_out.bias / 2)
            
            
            prediction = softmax(out)
            pred_v = softmax(out_v)
            pred_a = softmax(out_a)  
            for i in range(image.shape[0]):

                ma = np.argmax(prediction[i].cpu().data.numpy())
                v = np.argmax(pred_v[i].cpu().data.numpy())
                a = np.argmax(pred_a[i].cpu().data.numpy())
                num[label[i]] += 1.0
 
                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0 
                if np.asarray(label[i].cpu()) == v:
                    acc_v[label[i]] += 1.0  
                if np.asarray(label[i].cpu()) == a:
                    acc_a[label[i]] += 1.0 
 
    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)

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

    # Generate a random seed
    args.random_seed = np.random.randint(0, 100000)
    setup_seed(args.random_seed)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')


    train_dataset = AVDataset(args, mode='train')
    test_dataset = AVDataset(args, mode='test')
    
    try:
        val_dataset = AVDataset(args, mode='val')
    except:
        print("===")
        val_dataset = test_dataset 

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=8, pin_memory=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=8, pin_memory=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=8, pin_memory=False)
    
 
    model = AVClassifier(args,n_classes=train_dataloader.dataset.class_number)
    model.apply(weight_init)
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.cuda() 

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    if args.train:
        if not os.path.isdir(args.ckpt_path):
            os.makedirs(args.ckpt_path) 
    # logger initialization 
    logger = init_logging(args.ckpt_path + f'/log_{args.train}.txt')
    logger.info(f"===>PID:{os.getpid()}, GPU:[{args.gpu_ids}]")
    logger.info(args)
    # Load Vocabulary

    if args.train:
        save_config(args, os.path.join(args.ckpt_path, "config.json"))

        best_acc = 0.0

        for epoch in range(args.epochs):

            logger.info('Epoch: {}: '.format(epoch))

            if args.use_tensorboard:

                writer_path = os.path.join(args.tensorboard_path, args.dataset)
                if not os.path.exists(writer_path):
                    os.mkdir(writer_path)
                log_name = '{}_{}'.format(args.fusion_method, args.modulation)
                writer = SummaryWriter(os.path.join(writer_path, log_name))

                batch_loss, batch_loss_f, batch_loss_a, batch_loss_v = train_epoch(args, epoch, model, device,
                                                                     train_dataloader, optimizer, scheduler)
                acc, acc_a, acc_v = valid(args, model, device, val_dataloader)

                writer.add_scalars('Loss', {'Total Loss': batch_loss,
                                            'Audio Loss': batch_loss_a,
                                            'Visual Loss': batch_loss_v}, epoch)

                writer.add_scalars('Evaluation', {'Total Accuracy': acc,
                                                  'Audio Accuracy': acc_a,
                                                  'Visual Accuracy': acc_v}, epoch)

            else:
                batch_loss, batch_loss_f, batch_loss_a, batch_loss_v = train_epoch(args, epoch, model, device,
                                                                     train_dataloader, optimizer, scheduler,logger=logger)
                
                acc, acc_a, acc_v = valid(args, model, device, val_dataloader)

            if acc > best_acc:
                best_acc = float(acc)

                if not os.path.exists(args.ckpt_path):
                    os.mkdir(args.ckpt_path)

                model_name = 'model_best.pth'

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
                # logger.info("Loss: {:.3f}, Acc: {:.3f}".format(batch_loss, acc))
                logger.info("Loss: {:.3f}, Loss_a: {:.3f}, Loss_v: {:.3f}, Loss_f: {:.3f}, Best Acc: {:.3f}".format(batch_loss, batch_loss_a, batch_loss_v,batch_loss_f, best_acc))
                logger.info("Audio Acc: {:.3f}, Visual Acc: {:.3f} Fusion Acc: {:.3f}".format(acc_a, acc_v, acc))
            else:
                # logger.info("Loss: {:.3f}, Acc: {:.3f}, Best Acc: {:.3f}".format(batch_loss, acc, best_acc))

                logger.info("Loss: {:.3f}, Loss_a: {:.3f}, Loss_v: {:.3f}, Loss_f: {:.3f}, Best Acc: {:.3f}".format(batch_loss, batch_loss_a, batch_loss_v,batch_loss_f, best_acc))
                logger.info("Audio Acc: {:.3f}, Visual Acc: {:.3f} Fusion Acc: {:.3f}".format(acc_a, acc_v, acc))
            
            # Save the last model at the end of each epoch
            if not os.path.exists(args.ckpt_path):
                os.mkdir(args.ckpt_path)
            last_model_name = 'model_last.pth'
            last_saved_dict = {'saved_epoch': epoch,
                          'modulation': args.modulation,
                          'alpha': args.alpha,
                          'fusion': args.fusion_method,
                          'acc': acc,
                          'model': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'scheduler': scheduler.state_dict()}
            last_save_dir = os.path.join(args.ckpt_path, last_model_name)
            torch.save(last_saved_dict, last_save_dir)

    else:
        # load and test best model
        logger.info('================ Testing Best Model ================')
        loaded_dict = torch.load(args.ckpt_path+'/model_best.pth')
        # epoch = loaded_dict['saved_epoch']
        modulation = loaded_dict['modulation']
        # alpha = loaded_dict['alpha']
        fusion = loaded_dict['fusion']
        state_dict = loaded_dict['model']

        assert modulation == args.modulation, 'inconsistency between modulation method of loaded model and args !'
        assert fusion == args.fusion_method, 'inconsistency between fusion method of loaded model and args !'

        model.load_state_dict(state_dict)
        logger.info('Best trained model loaded!')

        acc, acc_a, acc_v = valid(args, model, device, test_dataloader)
        logger.info('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))

        # load and test last model
        logger.info('================ Testing Last Model ================')
        try:
            loaded_dict_last = torch.load(args.ckpt_path+'/model_last.pth')
            model.load_state_dict(loaded_dict_last['model'])
            logger.info('Last trained model loaded!')

            acc_last, acc_a_last, acc_v_last = valid(args, model, device, test_dataloader)
            logger.info('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc_last, acc_a_last, acc_v_last))
        except FileNotFoundError:
            logger.info('Last model not found.')


if __name__ == "__main__":
    main()
