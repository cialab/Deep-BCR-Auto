import argparse
import pandas as pd
import torch, torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from ctran import ctranspath
import os
import glob
from os.path import join
import numpy as np
import openslide


parser = argparse.ArgumentParser(description='outliers data preprocessing')
parser.add_argument('--save', default='../data/feats/cam17CTP', type=str, help='Saving directory')
parser.add_argument('--pts', default='/isilon/datalake/cialab/scratch/cialab/Ziyu/attention2minority/data/pts/cam17l1p224s224/', type=str, help='Data directory')
parser.add_argument('--label', default='../reference17.csv', type=str, help='label file')
parser.add_argument('--cohort', default='*', type=str, help='training or testing cohort')
parser.add_argument('--start', default=0, type=int, help='start WSI')
parser.add_argument('--end', default=None, type=int, help='end WSI')

args = parser.parse_args()
if not os.path.exists(args.save):
    os.mkdir(args.save)
    os.mkdir(join(args.save, 'train'))
    os.mkdir(join(args.save, 'train', 'normal'))
    os.mkdir(join(args.save, 'train', 'tumor'))
    os.mkdir(join(args.save, 'test'))
    os.mkdir(join(args.save, 'test', 'normal'))
    os.mkdir(join(args.save, 'test', 'tumor'))

labeldf = pd.read_csv(args.label, index_col=0, header=None)


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
trnsfrms_val = transforms.Compose(
    [
        transforms.Resize(224, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std)
    ]
)
# class roi_dataset(Dataset):
#     def __init__(self, img_csv,
#                  ):
#         super().__init__()
#         self.transform = trnsfrms_val

#         self.images_lst = img_csv

#     def __len__(self):
#         return len(self.images_lst)

#     def __getitem__(self, idx):
#         path = self.images_lst.filename[idx]
#         image = Image.open(path).convert('RGB')
#         image = self.transform(image)


#         return image

def main(args=args):
    namelist = sorted(glob.glob('/isilon/datalake/cialab/original/cialab/image_database/d00142/CAMELYON17/{}/*/*/*.tif'.format(args.cohort)))

    model = ctranspath()
    model.head = nn.Identity()
    td = torch.load(r'./ctranspath.pth')
    model.load_state_dict(td['model'], strict=True)

    model.cuda()
    model.eval()
    with torch.no_grad():
        for name in namelist[args.start:args.end]:
            train, label = assign(labeldf, name)
            pid = name.split('/')[-1].split('.')[0]
            print('**********')
            print(pid)
            pts = np.load(join(args.pts, pid+'.npy'))
            print('{} patches'.format(pts.shape[0]))

            with openslide.OpenSlide(name) as fp:
                dataloader = load_dataset(fp, pts)
                feats = np.empty((0,768), dtype='float32')
                for images in dataloader:
                    images = images.cuda()

                    z1 = model(images)
                    z1 = z1.cpu().numpy()

                    feats.resize((feats.shape[0]+z1.shape[0], 768))
                    feats[-z1.shape[0]:] = z1

            save_dir = join(args.save, train, label, pid+'.npy')
            np.save(save_dir, feats)
            print('{}x{} bag saved in '.format(feats.shape[0], feats.shape[1]), save_dir)


class cam17P(Dataset):
    def __init__(self, fp, pts, transform=None):
        
        self.pts = pts
        self.fp = fp
        self.transform = transform

    def __len__(self):
        return len(self.pts)

    def __getitem__(self, idx):
        pt = self.pts[idx]
        image = self.fp.read_region((pt[1], pt[0]), 0, (256, 256)).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        return image

def assign(labeldf, name):
    if labeldf.loc[name.split('/')[-1].split('.')[0]][1] == 'Normal':
        label = 'normal'
    elif labeldf.loc[name.split('/')[-1].split('.')[0]][1] == 'Tumor':
        label = 'tumor'
    if name.split('/')[-4] == 'training':
        train = 'train'
    elif name.split('/')[-4] == 'testing':
        train = 'test'
    else:
        raise ValueError('Wrong category')
    return train, label

def load_dataset(fp, pts):
    test_dataset = cam17P(fp=fp, pts=pts, transform=trnsfrms_val)
    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=50, shuffle=False,
            num_workers=16, pin_memory=True)
    
    return test_loader

if __name__ == '__main__':
    main()