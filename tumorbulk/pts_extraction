import openslide
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.color import rgb2gray
import glob
import os
from os.path import join
import argparse
from tqdm import tqdm
import pandas as pd


parser = argparse.ArgumentParser(description='save patch coordinates into npy. Coords under certain level 0. \
                                                (IN (row, col) INDEXING, NOT (x, y))')
parser.add_argument('-p', default=448, type=int, help='Patch size')
parser.add_argument('-s', default=448, type=int, help='Stride')
parser.add_argument('-l', default=0, type=int, help='Magnification level')
parser.add_argument('--save', default='../casii/data/', type=str, help='Saving directory')
parser.add_argument('--datadf', default='../tcga_brca_bulkdetected.csv', 
                    type=str, help='dataset excel')
parser.add_argument('--datadir', type=str, default='/isilon/datalake/cialab/original/cialab/image_database/d00142/tcga_brca_latest',
                        help='slide dir')                   
parser.add_argument('--code', default='TCGA', type=str, help='code')    
parser.add_argument('--start', default=0, type=int, help='start')                                  


args = parser.parse_args()

def main(args=args):
    filedf = pd.read_csv(args.datadf, header=0, index_col='slide')

    namelist = glob.glob(join(args.datadir, '*', '*.svs'))
    foldermap = {}
    for name in namelist:
        pid = os.path.splitext(os.path.basename(name))[0]
        foldermap[pid] = name

    save_dir = join(args.save, 'pts', args.code+'l'+str(args.l)+'p'+str(args.p)+'s'+str(args.s))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    maskdir = join('./results', args.code, 'masks')

    for pid in filedf.index[args.start:]:
        dataname = foldermap[pid] 
        # read mask
        pid = os.path.basename(dataname).split('.svs')[0]
        maskname = join(maskdir, f'{pid}.png')
        if not os.path.exists(maskname):
            print('MASK NOT EXISTS')
            break 

        mask = cv2.imread(maskname, 0)

        print('***********')
        print(pid)

        # load wsi
        with openslide.OpenSlide(dataname) as fp:
            w, h = fp.level_dimensions[args.l]
            w0, h0 = fp.dimensions

            #extract coords in (x, y)
            pts = extract(fp, w, h, w0, h0, args.p, args.s, mask, args.l)


        # save pts
        np.save(join(save_dir, pid+'.npy'), pts)



def extract(fp, w, h, w0, h0, ps, stride, mask, level):

    boundaries = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    boundaries = boundaries[0]
    minw, minh, maxw, maxh = mask.shape[1], mask.shape[0], 0, 0
    for i in range(len(boundaries)):
        b = np.squeeze(boundaries[i], 1)
        if np.min(b[:, 0]) < minw:
            minw = np.min(b[:, 0])
        if np.min(b[:, 1]) < minh:
            minh = np.min(b[:, 1])
        if np.max(b[:, 0]) > maxw:
            maxw = np.max(b[:, 0])
        if np.max(b[:, 1]) > maxh:
            maxh = np.max(b[:, 1])

    psy = ps * mask.shape[0] / float(h)
    psx = ps * mask.shape[1] / float(w)

    stride = stride * mask.shape[0] / float(h)

    # Grid of points
    ys = np.arange(minh, maxh, stride)
    xs = np.arange(minw, maxw, stride)

    [ys, xs] = np.meshgrid(ys, xs, indexing='ij')
    ys = ys.reshape((-1, 1))
    xs = xs.reshape((-1, 1))
    pts = np.concatenate([ys, xs], 1)

    # Here's where we put things
    bag = np.zeros((ps, ps, 3, pts.shape[0]), 'uint8')
    keep = np.zeros((pts.shape[0],), dtype=bool)

    for p in tqdm(range(pts.shape[0])):
        # Query pts
        rx = pts[p, 1]
        ry = pts[p, 0]

        # Pass first one
        if p == 0:
            continue
        

        # Checks if inside the mask image
        if mask.shape[1] > rx + psx and mask.shape[0] > ry + psy and \
        np.mean(mask[int(ry):int(ry+psy), int(rx):int(rx+psx)] == 255) > 0.5:      
             
            im = np.array(fp.read_region((round(rx*float(w0)/mask.shape[1]), round(ry*float(h0)/mask.shape[0])), level, (ps, ps)).convert('RGB'))
            
            if np.mean(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) < 240) > 0.75:
                pts[p] = np.array([round(ry*float(h0)/mask.shape[0]), round(rx*float(w0)/mask.shape[1])])

                keep[p] = True

    print('{} patches in bounding box'.format(pts.shape[0]))
    pts = pts[keep]
    ptsxy = pts[:, ::-1] # Very important. save pts in (x, y) format
    print('Found {} tissue patches'.format(ptsxy.shape[0]))

    return np.around(ptsxy).astype('int')

if __name__ == '__main__':
    main()
