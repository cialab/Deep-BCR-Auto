import argparse
import os
from os.path import join
import glob
import numpy as np
import pandas as pd
import h5py

parser = argparse.ArgumentParser(description='lung')
parser.add_argument('--datadf', default='../tcga_brca_bulkdetected.csv', 
                    type=str, help='dataset excel')
parser.add_argument('--featdir', default='./data/ctpnormembedding/l0p896s896', type=str,
                    help='path to the feats of normal WSIs in trainng set')
parser.add_argument('--task', default='fivefold', type=str,
                    help='task')
parser.add_argument('--encoder', default='ctpnorm', type=str,
                    help='encoder name')
parser.add_argument('--fold', default=0, type=int,
                    help='fold index')                    
parser.add_argument('--savedir', default='./data/keys', type=str,
                    help='path to save learned key set')
parser.add_argument('--curdir', default='./data/cur', type=str,
                    help='path to save learned key set')
parser.add_argument('-t', default=100, type=int,
                    help='maximum number of keys from each normal WSI')       
parser.add_argument('--dim', default=768, type=int,
                    help='feat dim')  
parser.add_argument('--psize', default=448, type=int,
                    help='patch size')
parser.add_argument('--cur', action='store_true', default=False, 
                    help='run cur and save col ranking')  
parser.add_argument('--extract', action='store_true', default=False, 
                    help='extract keys based on cur results')         



def extractTopKColumns(matrix):
    '''
    Learn representative negative instances from each normal WSI
    '''
    score  = {}
    rank = np.linalg.matrix_rank(matrix)
    u, s, vh = np.linalg.svd(matrix, full_matrices=True)
    
    for j in range(0, matrix.shape[1]):
        cscore = sum(np.square(vh[0:rank,j]))
        cscore /= rank
        score[j] = min(1, rank*cscore)
        
    prominentColumns = sorted(score, key=score.get, reverse=True)[:rank]
    #Removal of extra dimension\n",
    C = np.squeeze(matrix[:, [prominentColumns]])
    
    return ({"columns": prominentColumns, "matrix": C, "scores": sorted(score.values(), reverse = True)[:rank]})

def extract(choices, traindf, featdir, curdir, t, keyset):
    for index in choices:
        print(index)
        basename = index
        filename = join(featdir, basename+'.npy')
        curname = join(curdir, basename+'.npy')

        feats = np.load(filename).T
        cols = np.load(curname)
        keys = np.transpose(np.squeeze(feats[:, cols])) #back to n x dim
        

        length = keys.shape[0]

        if length <= t:
            keyset = np.vstack([keyset, keys])
        else:
            keyset = np.vstack([keyset, keys[:t]])

        print(keyset.shape)

    return keyset

def run(args):
    sizecode = args.featdir.split('/')[-1] # indicate the size, stride, level of our patch
    if args.cur:
        # save all CUR results for all slides
        if not os.path.exists(join(args.curdir, args.task)):
            os.mkdir(join(args.curdir, args.task))
        if not os.path.exists(join(args.curdir, args.task, sizecode)):
            os.mkdir(join(args.curdir, args.task, sizecode))

        filedf = pd.read_csv(args.datadf, index_col='slide')
        for index in filedf.index:
            basename = index
            filename = join(args.featdir, basename+'.npy')
            feats = np.load(filename).T
            res = extractTopKColumns(feats)
            cols = res["columns"]

            np.save(join(args.curdir, args.task, sizecode, f"{basename}.npy"), cols)

    if args.extract:
        nfold = len(glob.glob('./splits/{}/splits_*.csv'.format(args.task)))
        for i in range(2,3):
            print(f'***fold {i}')
            args.fold = i

            # prepare keyset file
            if not os.path.exists(join(args.savedir, args.task)):
                os.mkdir(join(args.savedir, args.task))

            split = pd.read_csv('./splits/{}/splits_{}.csv'.format(args.task, args.fold), header=0)
            filedf = pd.read_csv(args.datadf, index_col='slide')
            trainindex = split['train'].dropna()

            traindf = filedf.loc[trainindex]
            highchoices = traindf.loc[traindf['odx85'] == 'H'].sample(n=35, random_state=42).index
            lowchoices = traindf.loc[traindf['odx85'] == 'L'].sample(n=35, random_state=42).index


            highkeyset = np.empty((0, args.dim))
            lowkeyset = np.empty((0, args.dim))

            highkeyset = extract(highchoices, traindf, args.featdir, join(args.curdir, args.task, sizecode), args.t, highkeyset)
            lowkeyset = extract(lowchoices, traindf, args.featdir, join(args.curdir, args.task, sizecode), args.t, lowkeyset)
            
            np.save(join(args.savedir, args.task, '{}-p{}-high-{}-f{}'.format(args.encoder, args.psize, args.t, args.fold)), highkeyset)
            np.save(join(args.savedir, args.task, '{}-p{}-low-{}-f{}'.format(args.encoder, args.psize, args.t, args.fold)), lowkeyset)

if __name__ == '__main__':
    args = parser.parse_args()

    run(args)
