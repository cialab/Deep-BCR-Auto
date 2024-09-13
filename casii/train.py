import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

import argparse
import os
import math
import random
import time
import mymodel.mymodel as model
import numpy as np
import glob
import shutil
import pandas as pd
from PIL import Image
import mydataset.mydataset as mydataset
import utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser(description='tcga brca odx prediction')
parser.add_argument('--model', default='', type=str,
                    help='path to the model')
parser.add_argument('--arch', default='CASii_MB', type=str,
                    help='architecture')
parser.add_argument('--data', default='fivefold', type=str,
                    help='dataset')  
parser.add_argument('--psize', default=896, type=int,
                    help='patch size') 
parser.add_argument('-t', default=100, type=int,
                    help='CUR top-t')  
parser.add_argument('--seed', default=7, type=int,
                    help='torch random seed')                
parser.add_argument('--split', default=42, type=int,
                    help='split random seed')                  
parser.add_argument('--nfold', default=5, type=int,
                    help='num of folds')
parser.add_argument('--startfold', default=42, type=int,
                    help='num of folds')                    
parser.add_argument('-b', '--batch_size', default=1, type=int,
                    help='batch size')          
parser.add_argument('--epochs', default=20, type=int,
                    help='number of epochs')   
parser.add_argument('--inputd', default=768, type=int,
                    help='input dim')         
parser.add_argument('--hd', default=384, type=int,
                    help='hidden layer dim')                                
parser.add_argument('--code', default='test', type=str,
                    help='exp code') 
parser.add_argument('--encoder', default='ctpnorm', type=str,
                    help='encoder')  


parser.add_argument('--lr', default=2e-4, type=float, 
                    help='init learning rate')   
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--weighted_sample', action='store_true', 
                    default=False, help='enable weighted sampling')                    
parser.add_argument('--freq', default=100, type=int, 
                    help='training log display frequency')                              
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')  
parser.add_argument('--pretrained', default='', type=str, 
                    help='pretrained model for validate')    

parser.add_argument('--patience', default=5, type=int, 
                    help='early stopping patience') 
parser.add_argument('--stop_epoch', default=5, type=int, 
                    help='start epoch to activate early stopping counting') 
parser.add_argument('--monitor', default='loss', type=str, 
                    help='value to monitor for early stopping')                                                                                             

# parser.add_argument('-k', default=10, type=float, help='top-k query predictions')
# parser.add_argument('--tau', default=1, type=float, help='softmax temperature')
# parser.add_argument('--keys', default='', type=str, help='keyset file')
# parser.add_argument('--sc', default=None, type=float, help='use sparse coding')
# parser.add_argument('--ql1', default=None, type=float, help='use botk query loss')
# parser.add_argument('--ql2', default=None, type=float, help='use topk query loss')

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def run(args):

    train_dataset = getattr(mydataset, args.data)(train='train', args=args, split=args.split)

    if args.weighted_sample:
        weights = train_dataset.get_weights()
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=torch.utils.data.WeightedRandomSampler(weights, len(weights)),
            num_workers=32, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=32, pin_memory=True)

    val_dataset = getattr(mydataset, args.data)(train='val', args=args, split=args.split)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=32, pin_memory=True)

    test_dataset = getattr(mydataset, args.data)(train='test', args=args, split=args.split)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=32, pin_memory=True)

    # Get FA layer dimensions, [lowkeydim, highkeydim]
    A_dims = train_dataset.get_keysetdims()

    # Define CASii-MB model
    net = getattr(model, args.arch)(inputd=args.inputd, hd=args.hd, n_head=2, A_dims=A_dims)

    # BCE loss, Adam opt
    criterions = [nn.CrossEntropyLoss().cuda('cuda')]

    parameters = list(filter(lambda p: p.requires_grad, net.parameters()))

    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.reg)


    net.cuda()
    writer = SummaryWriter(os.path.join(args.save, time.strftime("%Y-%m-%d-%H:%M:%S", time.gmtime())))
    
    # set-up early stopping
    EarlyStopping = utils.EarlyStopping(save_dir=args.save, args=args)
    monitor_values = {'acc':0, 'auc':1, 'loss':4}
    monitor_idx = monitor_values[args.monitor]
    for epoch in range(args.epochs):

        # train for one epoch
        train(train_loader, net, criterions, optimizer, epoch, args, writer)

        # evaluate on validation set (acc, auc, sen, spe, loss)
        metrics = validate(val_loader, net, epoch, criterions, args, writer, 'val')

        # early stopping
        EarlyStopping(epoch, metrics[monitor_idx], net, optimizer)

        # evaluate on testing set
        if EarlyStopping.early_stop:
            _ = validate(test_loader, net, epoch, criterions, args, writer, 'test')
            print('****Early stop at epoch:{}'.format(epoch-args.patience))
            break
        else:
            if EarlyStopping.counter == 0:
                best_metrics = validate(test_loader, net, epoch, criterions, args, writer, 'test')
                best_epoch = epoch
            else:
                _ = validate(test_loader, net, epoch, criterions, args, writer, 'test')

    print('****best testing result: epoch: {}, acc: {}, auc: {}, sen: {}, spe: {}, loss: {}'.format(best_epoch, best_metrics[0], best_metrics[1], \
    best_metrics[2], best_metrics[3], best_metrics[4]))

def train(train_loader, model, criterions, optimizer, epoch, args, writer):
    criterion = criterions[0]
    losses = utils.AverageMeter('Loss', ':.4e')

    progress = utils.ProgressMeter(
        len(train_loader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    for i, data in enumerate(train_loader):
        images, highkeys, lowkeys, target = data
        images = images.cuda()
        highkeys, lowkeys = highkeys.cuda(), lowkeys.cuda()
        target = target.cuda()
        Y_hat, output, _ = model((images, lowkeys, highkeys))

        if i == 0:
            outputs = output
            targets = target
        else:
            outputs = torch.cat((outputs, output), 0)
            targets = torch.cat((targets, target), 0)

        loss = criterion(output, target)

        
        losses.update(loss.item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.freq == 0:
            progress.display(i)

    acc, sen, spe = utils.accuracy(outputs, targets, False)

    if writer:
        writer.add_scalar("Loss/train", losses.avg, epoch)
        writer.add_scalar("Accuracy/train", acc, epoch)
        writer.add_scalar("sen/train", sen, epoch)
        writer.add_scalar("spe/train", spe, epoch)


def validate(val_loader, model, epoch, criterions, args, writer, val='val'):
    criterion = criterions[0]
    losses = utils.AverageMeter('Loss', ':.4e')

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):    
            images, highkeys, lowkeys, target = data
            images = images.cuda()
            highkeys, lowkeys = highkeys.cuda(), lowkeys.cuda()
            target = target.cuda()
            Y_hat, output, _ = model((images, lowkeys, highkeys))
            # output = output.view(-1).float()

            if i == 0:
                outputs = output
                targets = target
            else:
                outputs = torch.cat((outputs, output), 0)
                targets = torch.cat((targets, target), 0)

            loss = criterion(output, target)
            losses.update(loss.item(), images.size(0))

    acc, sen, spe, auc = utils.accuracy(outputs, targets)

    if val == 'val':
        print(' **Validation Acc {acc:.3f} sen {sen:.3f} spe {spe:.3f} AUC {auc:.3f} LOSS {loss:.3f}'
            .format(acc=acc, sen=sen, spe=spe, auc=auc, loss=losses.avg))
    else:
        print(' ***Testing Acc {acc:.3f} sen {sen:.3f} spe {spe:.3f} AUC {auc:.3f} LOSS {loss:.3f}'
            .format(acc=acc, sen=sen, spe=spe, auc=auc, loss=losses.avg))

    if writer:
        writer.add_scalar("Loss/"+val, losses.avg, epoch)
        writer.add_scalar("Accuracy/"+val, acc, epoch)
        writer.add_scalar("sen/"+val, sen, epoch)
        writer.add_scalar("spe/"+val, spe, epoch)
        writer.add_scalar("auc/"+val, auc, epoch)

    return acc, auc, sen, spe, losses.avg




if __name__ == '__main__':
    args = parser.parse_args()
    
    
    if 'ctp' in args.code:
        save_dir = './{}results/{}_{}_s{}_p{}'.format(args.encoder, args.data, args.code, args.seed, args.psize)
    else:
        save_dir = './{}results/{}_{}_s{}_p{}'.format(args.encoder, args.data, args.code, args.seed, args.psize)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)    

    for i in range(args.startfold, args.startfold+args.nfold):     
        seed_torch(args.seed)  
        args.split = i       
        args.save = os.path.join(save_dir, str(args.split))                  
        run(args)