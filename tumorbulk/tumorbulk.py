import mymodel.model as model
import openslide
import h5py
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import torchstain
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from skimage.color import rgb2gray
from skimage.transform import rescale
import glob
import os
from os.path import join
from tqdm import tqdm

from skimage.morphology import (erosion, binary_closing, binary_opening)
from skimage.morphology import convex_hull_object, remove_small_objects
from skimage.morphology import disk
from skimage.measure import regionprops, label

import argparse

CONFIG = {'fsize': 10, 'remove': 5000, 'convex': True, 'keep': 2}


def areafilter(mask, keep):
    masklabel = label(mask)
    rp = regionprops(masklabel)
    rpareas = [r.area for r in rp]
    rpareas = sorted(rpareas)
    thre = rpareas[-min(keep, len(rpareas))]
    
    for i in range(min(keep, len(rpareas))+1, len(rpareas)+1):
        if rpareas[-i] >= 0.9*thre:
            thre = rpareas[-i]
        else:
            break

    return remove_small_objects(mask, thre)

def refinemask(mask, fsize=10, remove=5000, convex=True, keep=2):
    mask_b = mask > 0.5
    footprint = disk(fsize)

    mask_b = binary_closing(mask_b, disk(5))
    mask_b = binary_opening(mask_b, footprint)
    mask_b = remove_small_objects(mask_b, remove)
    if convex:
        mask_b = convex_hull_object(mask_b)

    mask_b = areafilter(mask_b, keep)

    return np.uint8(mask_b) * 255

class dataset(Dataset):
    def __init__(self, pts, fp, level, ps, normalizer=None):
        self.pts = pts
        self.fp = fp
        self.level = level
        self.ps = ps
        self.normalizer = normalizer
        if normalizer:
            self.normtransf = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x*255)
                                ])
            self.transform = transforms.Compose(
                                        [
                                            transforms.Lambda(lambda x: x/255),
                                            transforms.Resize(224),
                                            transforms.Normalize(mean = (0.485, 0.456, 0.406), 
                                            std = (0.229, 0.224, 0.225))
                                        ]
                                    )
        else:
            self.transform = transforms.Compose(
                                        [
                                            transforms.Resize(224), #for ctranspath
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                        ]
                                    )
                                            
    def __len__(self):
        return self.pts.shape[0]
    
    def __getitem__(self, idx):
        rx = self.pts[idx, 0]
        ry = self.pts[idx, 1]
        image = self.fp.read_region((rx, ry), self.level, (self.ps, self.ps)).convert('RGB')
        if self.normalizer:
            image = self.normtransf(image)
            image = self.normalizer.normalize(I=image)
            image = torch.permute(image, (2, 0, 1))

        image = self.transform(image)
        
        return image
    
def predict_(net, im):
    predict = torch.sigmoid(net(im).view(-1).float()).cpu().detach().numpy()
    return predict

def predict(net, normalizer, fp, ptsdir, pid, level=0, ps=448):
    # TODO: find a universal level for mask 
    w, h = fp.level_dimensions[min(fp.level_count-1, 3)]
    mask = np.zeros([h, w])
    w0, h0 = fp.dimensions
    xratio, yratio = w/w0, h/h0
    pt_h5name = join(ptsdir, pid+'.h5')
    with h5py.File(pt_h5name, 'r') as f:
        pts = f['coords'][:]

    with torch.no_grad(): 
        # build pts dataset and dataloader  
        ptsdataset = dataset(pts, fp, level, ps, normalizer)
        dataloader = torch.utils.data.DataLoader(
            ptsdataset, batch_size=20, shuffle=False,
            num_workers=32, pin_memory=True)
        predicts = []

        # predict patches in batches
        for i, im in enumerate(tqdm(dataloader)):
            im = im.cuda()
            predict = predict_(net, im)
            predicts.append(predict) 

        predicts = np.hstack(predicts)

        # map predicts into mask
        for i in tqdm(range(pts.shape[0])):
            rx = pts[i, 0]
            ry = pts[i, 1]

            mx, my, mpsx, mpsy = int(rx*xratio), int(ry*yratio), int(ps*xratio), int(ps*yratio)

            mask[my:my+mpsy, mx:mx+mpsx] = predicts[i]

    return mask

def overlay(fp, mask, masklevel=-1):
    # TODO: find a universal level for overlay 
    w, h = fp.level_dimensions[min(fp.level_count-1, 3)]
    overlay = fp.read_region((0, 0), min(fp.level_count-1, 3), (w, h)).convert('RGB')

    draw = np.array(overlay)
    # Find Canny edges
    edged = cv2.Canny(mask, 30, 200)

    # Finding Contours
    # Use a copy of the image e.g. edged.copy()
    # since findContours alters the image
    contours, hierarchy = cv2.findContours(edged, 
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(draw, contours, -1, (0, 255, 0), 20)
    
    return draw

def init_normalizer():
    template = cv2.cvtColor(cv2.imread("./template.jpg"), cv2.COLOR_BGR2RGB)
    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*255)
    ])


    normalizer = torchstain.normalizers.ReinhardNormalizer(backend='torch')
    normalizer.fit(T(template))

    return normalizer

def run(args):
    # load slide names, build folder map
    # TODO: changed to .svs. Needs to generalize to multiple file types in the future
    namelist = glob.glob(join(args.datadir, '*', '*.svs'))
    foldermap = {}
    for name in namelist:
        pid = os.path.splitext(os.path.basename(name))[0]
        foldermap[pid] = name

    # Load tumor classification model (CTransPath)
    net = model.ctrans()
    print('load model from: ', 'model_best.pth.tar')
    checkpoint = torch.load('model_best.pth.tar', map_location="cpu")

    state_dict = checkpoint['state_dict']
    msg = net.load_state_dict(state_dict, strict=True)
    print(msg.missing_keys)
    net.cuda()
    net.eval()

    # init reinhard normalizer
    normalizer = init_normalizer()

    # build missing log
    if os.path.exists(join(args.savedir, args.code, 'missing.csv')):
        raise FileExistsError("Missing log already exists. Rename the existing one to keep it.")

    missingdf = pd.DataFrame({'ID': []})
    missingdf.to_csv(join(args.savedir, args.code, 'missing.csv'))

    if not os.path.exists(join(args.savedir, args.code)):
        os.mkdir(join(args.savedir, args.code))
        os.mkdir(join(args.savedir, args.code, 'masks'))
        os.mkdir(join(args.savedir, args.code, 'visual'))

    
    for pid in foldermap:
        filename = foldermap[pid]
        if not os.path.exists(join(args.ptsdir, pid+'.h5')) or \
        os.path.exists(join(args.savedir, args.code, 'masks', f'{pid}.png')): # skip if coordinates file not exists or mask already exists
            print(f'***************SKIP {pid}***************')
            continue

        print('slide name: ', pid)
        with openslide.OpenSlide(filename) as fp:
            try:
                mask = predict(net, normalizer, fp, args.ptsdir, pid, args.level, args.psize)

                refinedmask = refinemask(mask, **CONFIG) 

                cv2.imwrite(join(args.savedir, args.code, 'masks', pid+'.png'), refinedmask)

                draw = overlay(fp, refinedmask)
                drawlr = (rescale(draw[:, :], 0.8, channel_axis=2) * 255).astype('uint8')
                cv2.imwrite(join(args.savedir, args.code, 'visual', pid+'.png'), drawlr)
            except:
                missing = pd.DataFrame({'ID': [pid]})
                missingdf = pd.concat([missingdf, missing], axis=0)
                missingdf.to_csv(join(args.savedir, args.code, 'missing.csv'))

                print('================MISSING {}================'.format(pid))
                continue



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create tumor mask')
    parser.add_argument('--datadir', type=str, default='/isilon/datalake/cialab/original/cialab/image_database/d00142/tcga_brca_latest',
                        help='slide dir')
    parser.add_argument('--ptsdir', type=str, default='../CLAM/results/TCGA/patches',
                        help='pts dir')
    parser.add_argument('--savedir', type=str, default='./results',
                        help='save dir')
    parser.add_argument('--code', type=str, default='TCGA',
                        help='running code')
    parser.add_argument('--level', type = int, default=0,
                        help='slide level')
    parser.add_argument('--psize', type=int, default=448,
                        help='patch size')
    
    args = parser.parse_args()

    run(args)
