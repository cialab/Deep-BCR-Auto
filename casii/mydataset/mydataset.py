import numpy as np
import glob
import os
from os.path import join
import random
import h5py
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

class fivefold(Dataset):
    '''
    dataset for lung carcinoma pilot study data. predict recur or not in 5 yrs
    '''
    def __init__(self, train='train', transform=None, args=None, split=42):

        self.img_dir = './data/{}embedding/l0p{}s{}'.format(args.encoder, args.psize, args.psize)

        self.split = pd.read_csv(join('./splits/fivefold', 'splits_{}.csv'.format(split-42)), header=0)
        self.labels = pd.read_csv('../tcga_brca_bulkdetected.csv', index_col='slide')
        self.highkeys = np.load('./data/keys/fivefold/{}-p{}-high-{}-f{}.npy'.format(args.encoder, args.psize, args.t, split-42))
        self.lowkeys = np.load('./data/keys/fivefold/{}-p{}-low-{}-f{}.npy'.format(args.encoder, args.psize, args.t, split-42))
        self.train = train
        
        if train == 'train':
            self.img_names = self.split.loc[:, ['train']].dropna()
        elif train == 'test':
            self.img_names = self.split.loc[:, ['test']].dropna()
        elif train == 'val':
            self.img_names = self.split.loc[:, ['val']].dropna()
            
        self.transform = transform

    def __len__(self):
        return len(self.img_names)
    
    def get_weights(self):
        # get weights for weight random sampler (training only)
        if self.train != 'train':
            raise TypeError('WEIGHT SAMPLING FOR TRAINING SET ONLY')
        N = len(self.img_names)
        w_per_cls = {'H': N/(self.split['train_label']=='H').sum(), 'L': N/(self.split['train_label']=='L').sum()}

        weights = [w_per_cls[self.labels.loc[name, 'odx85']] for name in self.img_names['train']]

        return torch.DoubleTensor(weights)
    
    def get_testnames(self):
        return self.split['test'].dropna().tolist()

    def get_keysetdims(self):
        return [self.lowkeys.shape[0], self.highkeys.shape[0]]

    def __getitem__(self, idx):
        img_path = join(self.img_dir, self.img_names.iloc[idx][self.train]+'.npy')
        image = np.load(img_path)

        label = int(self.labels.loc[self.img_names.iloc[idx][self.train], 'odx85'] == 'H')
            
        return torch.Tensor(image), torch.Tensor(self.highkeys), torch.Tensor(self.lowkeys), label



class threefold(Dataset):
    '''
    dataset for lung carcinoma pilot study data. predict recur or not in 5 yrs
    '''
    def __init__(self, train='train', transform=None, args=None, split=42):

        self.img_dir = './data/{}embedding/l0p{}s{}'.format(args.encoder, args.psize, args.psize)

        self.split = pd.read_csv(join('./splits/threefold', 'splits_{}.csv'.format(split-42)), header=0)
        self.labels = pd.read_csv('../tcga_brca_bulkdetected.csv', index_col='slide')
        self.highkeys = np.load('./data/keys/threefold/{}-p{}-high-{}-f{}.npy'.format(args.encoder, args.psize, args.t, split-42))
        self.lowkeys = np.load('./data/keys/threefold/{}-p{}-low-{}-f{}.npy'.format(args.encoder, args.psize, args.t, split-42))
        self.train = train
        
        if train == 'train':
            self.img_names = self.split.loc[:, ['train']].dropna()
        elif train == 'test':
            self.img_names = self.split.loc[:, ['test']].dropna()
        elif train == 'val':
            self.img_names = self.split.loc[:, ['val']].dropna()
            
        self.transform = transform

    def __len__(self):
        return len(self.img_names)
    
    def get_weights(self):
        # get weights for weight random sampler (training only)
        if self.train != 'train':
            raise TypeError('WEIGHT SAMPLING FOR TRAINING SET ONLY')
        N = len(self.img_names)
        w_per_cls = {'H': N/(self.split['train_label']=='H').sum(), 'L': N/(self.split['train_label']=='L').sum()}

        weights = [w_per_cls[self.labels.loc[name, 'odx85']] for name in self.img_names['train']]

        return torch.DoubleTensor(weights)
    
    def get_testnames(self):
        return self.split['test'].dropna().tolist()

    def get_keysetdims(self):
        return [self.lowkeys.shape[0], self.highkeys.shape[0]]

    def __getitem__(self, idx):
        img_path = join(self.img_dir, self.img_names.iloc[idx][self.train]+'.npy')
        image = np.load(img_path)

        label = int(self.labels.loc[self.img_names.iloc[idx][self.train], 'odx85'] == 'H')
            
        return torch.Tensor(image), torch.Tensor(self.highkeys), torch.Tensor(self.lowkeys), label

class threefoldtestosu(Dataset):
    '''
    dataset for lung carcinoma pilot study data. predict recur or not in 5 yrs
    '''
    def __init__(self, train='train', transform=None, args=None, split=42):

        self.img_dir = './data/{}embedding/l0p{}s{}'.format(args.encoder, args.psize, args.psize)

        self.split = pd.read_csv(join('./splits/threefoldtestosu', 'splits_{}.csv'.format(split-42)), header=0)
        self.labels = pd.read_excel('./dataset_csv/r21data_clean.xlsx', index_col='slide')
        self.highkeys = np.load('./data/keys/threefold/{}-p{}-high-{}-f{}.npy'.format(args.encoder, args.psize, args.t, split-42))
        self.lowkeys = np.load('./data/keys/threefold/{}-p{}-low-{}-f{}.npy'.format(args.encoder, args.psize, args.t, split-42))
        self.train = train
        
        if train == 'train':
            self.img_names = self.split.loc[:, ['train']].dropna()
        elif train == 'test':
            self.img_names = self.split.loc[:, ['test']].dropna()
        elif train == 'val':
            self.img_names = self.split.loc[:, ['val']].dropna()
            
        self.transform = transform

    def __len__(self):
        return len(self.img_names)
    
    def get_weights(self):
        # get weights for weight random sampler (training only)
        if self.train != 'train':
            raise TypeError('WEIGHT SAMPLING FOR TRAINING SET ONLY')
        N = len(self.img_names)
        w_per_cls = {'H': N/(self.split['train_label']=='H').sum(), 'L': N/(self.split['train_label']=='L').sum()}

        weights = [w_per_cls[self.labels.loc[name, 'odx85']] for name in self.img_names['train']]

        return torch.DoubleTensor(weights)
    
    def get_testnames(self):
        return self.split['test'].dropna().tolist()

    def get_keysetdims(self):
        return [self.lowkeys.shape[0], self.highkeys.shape[0]]

    def __getitem__(self, idx):
        img_path = join(self.img_dir, self.img_names.iloc[idx][self.train]+'.npy')
        image = np.load(img_path)

        label = int(self.labels.loc[self.img_names.iloc[idx][self.train], 'odx85'] == 'H')
            
        return torch.Tensor(image), torch.Tensor(self.highkeys), torch.Tensor(self.lowkeys), label