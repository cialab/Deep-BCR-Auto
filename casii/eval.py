import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


import argparse
import os
from os.path import join
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


parser = argparse.ArgumentParser(description='lung carcinoma pilot study recur or not in 5 yrs')
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
parser.add_argument('--epochs', default=100, type=int,
                    help='number of epochs')   
parser.add_argument('--inputd', default=768, type=int,
                    help='input dim')         
parser.add_argument('--hd', default=384, type=int,
                    help='hidden layer dim')                                
parser.add_argument('--code', default='test', type=str,
                    help='exp code') 
parser.add_argument('--encoder', default='ctpnorm', type=str,
                    help='encoder')   

parser.add_argument('--pretrained', default='model_best.pth.tar', type=str, 
                    help='pretrained model for validate')  

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

    # BCE loss, Adam opt
    criterions = [nn.CrossEntropyLoss().cuda('cuda')]

    val_dataset = getattr(mydataset, args.data)(train='val', args=args, split=args.split)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=32, pin_memory=True)

    test_dataset = getattr(mydataset, args.data)(train='test', args=args, split=args.split)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=32, pin_memory=True)

    A_dims = test_dataset.get_keysetdims()
    net = getattr(model, args.arch)(inputd=args.inputd, hd=args.hd, n_head=2, A_dims=A_dims)


    print('load model from: ', join(args.save, args.pretrained))
    checkpoint = torch.load(join(args.save, args.pretrained), map_location="cpu")

    state_dict = checkpoint['state_dict']
    msg = net.load_state_dict(state_dict, strict=True)
    print(msg.missing_keys)
    net.cuda()
    
    val_metrics = validate(val_loader, net, criterions, val='val')
    test_metrics = validate(test_loader, net, criterions, val='test')

    return test_metrics[-2:], val_metrics[-2:]


def validate(val_loader, model, criterions, val):
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


    return acc, auc, sen, spe, losses.avg, torch.softmax(outputs, dim = 1).cpu().numpy(), targets.cpu().numpy()


if __name__ == '__main__':
    args = parser.parse_args()
    seed_torch(args.seed)
    if 'ctp' in args.code:
        save_dir = './{}results/{}_{}_s{}_p{}'.format(args.encoder, args.data, args.code, args.seed, args.psize)
    else:
        save_dir = './{}results/{}_{}_s{}_p{}'.format(args.encoder, args.data, args.code, args.seed, args.psize)

    
    testresults = [['AUC', 'ACC', 'SPE', 'SEN', 'F1']]
    valresults = [['AUC', 'ACC', 'SPE', 'SEN', 'F1']]
    

    for i in range(42, 42+args.nfold): 
        testnamelist = pd.read_csv(join(f'./splits/{args.data}', 'splits_{}.csv'.format(i-42)), header=0)['test'].dropna().tolist()
        valnamelist = pd.read_csv(join(f'./splits/{args.data}', 'splits_{}.csv'.format(i-42)), header=0)['val'].dropna().tolist()
        testdf = {}
        valdf = {}     
        args.split = i       
        args.save = os.path.join(save_dir, str(args.split)) 

        test_results, val_results = run(args)

        testdf['Y'] = test_results[-1]
        valdf['Y'] = val_results[-1]

        testdf['p_0'] = test_results[-2][:, 0]
        valdf['p_0'] = val_results[-2][:, 0]

        testdf['p_1'] = test_results[-2][:, 1]
        valdf['p_1'] = val_results[-2][:, 1]

        testdf = pd.DataFrame(testdf, index=testnamelist)
        testdf.to_csv(join(save_dir, 'fold_{}.csv'.format(i-42)))

        valdf = pd.DataFrame(valdf, index=valnamelist)
        valdf.to_csv(join(save_dir, 'valfold_{}.csv'.format(i-42)))

        testresults.append(utils.eval_accuracy(testdf['p_1'], testdf['Y']))
        valresults.append(utils.eval_accuracy(valdf['p_1'], valdf['Y']))

    testresults = pd.DataFrame(testresults[1:], columns=testresults[0])
    valresults = pd.DataFrame(valresults[1:], columns=valresults[0])

    testresults.to_excel(join(save_dir, 'testsummary.xlsx'))
    valresults.to_excel(join(save_dir, 'valsummary.xlsx'))
    




    # for i in range(42, 42+args.nfold):       
    #     args.split = i       
    #     args.save = os.path.join(save_dir, str(args.split))                  
    #     test_metrics, val_metrics = run(args)
    #     aucs.append(test_metrics[0])
    #     accs.append(test_metrics[1])
    #     precisions.append(test_metrics[2])
    #     recalls.append(test_metrics[3])
    #     f1s.append(test_metrics[4])
    #     val_aucs.append(val_metrics[0])
    #     val_f1s.append(val_metrics[4])
    #     if i == 42:
    #         results['gts'] = test_metrics[-1]
    #     results[str(i)+'outputs'] = test_metrics[-2]
        
    # results = pd.DataFrame(results, index=testnamelist)
    # results.to_csv(join(save_dir, 'results.csv'))
    # print('')
    # print('========================================================')
    # print('*Val AUC {:.3f} +- {:.3f}'.format(np.mean(val_aucs), np.std(val_aucs)))
    # print('*Val f1 {:.3f} +- {:.3f}'.format(np.mean(val_f1s), np.std(val_f1s)))

    # print('**Testing AUC {:.3f} +- {:.3f}, ACC {:.3f} +- {:.3f}, PRECISION {:.3f} +- {:.3f}, RECALL {:.3f} +- {:.3f}, F1 {:.3F} +- {:.3F}'
    #     .format(np.mean(aucs), np.std(aucs), np.mean(accs), np.std(accs), np.mean(precisions), np.std(precisions), 
    #     np.mean(recalls), np.std(recalls), np.mean(f1s), np.std(f1s)))

    # bestidx = np.argmax(val_aucs)
    # print('***Best testing AUC {:.3f}, ACC {:.3f}, PRECISION {:.3f}, RECALL {:.3f}, F1 {:.3f}'
    #     .format(aucs[bestidx], accs[bestidx], precisions[bestidx], recalls[bestidx], f1s[bestidx]))

    # print('Saved in ', join(save_dir, 'results.csv'))