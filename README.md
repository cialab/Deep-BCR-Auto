# Deep-BCR-Auto
This is the official implementation of Deep-BCR-Auto.

## Tissue detection and patch cropping
1. Use any foreground segmentation method you like. Here, we use the [CLAM](https://github.com/mahmoodlab/CLAM/tree/master)'s implementation. Clone and go to the `/CLAM` folder.
2. Basic run (changed sthresh to 20):
```shell
python create_patches_fp.py --source <> --save_dir results/TCGA --patch_size 448 --step_size 448 --seg
```
3. Tune segmentations. Inspect each mask, tune `process_list_edited.csv` and set `processed` to 1 for the mask you want to tune:
```shell
python create_patches_fp.py --source <> --save_dir results/TCGA --patch_size 448 --step_size 448 --seg --process_list process_list_edited.csv
```
4. Get tissue patches. Set all `processed` to 1:
```shell
python create_patches_fp.py --source <> --save_dir results/TCGA --patch_size 448 --step_size 448 --seg --process_list process_list_edited.csv --patch --stitch
```

## Tumor bulk detection
1. GO TO   `/tumorbulk`
2. generate tumor bulk masks
```shell
python tumorbulk.py --datadir <> --ptsdir ../CLAM/results/TCGA/patches --savedir ./results --code TCGA
```
3. create a dataset excel based on those having masks. ('../tcga_brca_bulkdetected.csv')
4.  crop patches:
```shell
python pts_extraction.py -p 896 -s 896 -l 0 --code TCGA --datadf ../tcga_brca_bulkdetected.csv
```
Main args:
* `-p` : patch size
* `-s` : stride
* `-l` : patch extraction at which level (i.e., openslide downsample level)
* `--code` : experiment code
* `--datadf` : data spread sheet (i.e., dataset excel)

* ``**NOTE: pts are now saved in (x, y) format for consistency``

## Deep-BCR
1. go to `./casii/TransPath`.
2. encode patches:
```shell
python get_features_CTransPath_stainnorm.py --psize 896 --level 0 --datadf ../../tcga_brca_bulkdetected.csv --save ../data/ctpnormembedding --stainnorm
```
Main args:
* `--psize` : patch size
* `--stride` : stride
* `--level` : openslide level
* `--datadf` : dataset spread sheet. listed all the slide that you are using.
* `--stainnorm` : perform stain norm. store true, default: False
* `--save`: save dir

3. *(optional) encoding using resnet50:
```shell
python get_features_resnet_stainnorm.py --psize 896 --level 0 --datadf ../../tcga_brca_bulkdetected.csv --save ../data/resnormembedding --stainnorm 
```

3. build keyset:
```shell
python keyset_lrn.py --datadf '../tcga_brca_bulkdetected.csv' --featdir './data/ctpnormembedding/l0p896s896' --task fivefold -t 100 --psize 896
```
Main args:
* `--featdir` : feature dir
* `--task` : task name
* `-t` : maximum # of keys per slide
* `--cur` : run cur function. store true. 
* `--extract` : run keyset extraction function (must be after cur). store true. 
* `--psize` : patch size
* `--encoder` : encoder name. default: ctpnorm

4. run casii:
```shell
python train.py --arch CASii_MB --data threefold --code ctpnormTCGAodx_patience5_stopep10_ws_lr1e4 --psize 896 --nfold 3 --weighted-sample --patience 5 --stop_epoch 10 --lr 1e-4
```
Main args:
* `--arch` : model arch
* `--data` : dataset name in mydatasets
* `--ctp_recurnot` : exp code
* `--psize` : patch size
* `--weighted_sample` : store-true, default: False
* `--nfold` : num of folds, default: 5

5. test
```shell
python eval.py --arch CASii_MB --data threefold --code ctpnormTCGAodx_patience5_stopep10_ws_lr1e4 --psize 896 --nfold 3
```
Main args:
Main args:
* `--arch` : model arch
* `--data` : dataset name in mydatasets
* `--ctp_recurnot` : exp code
* `--psize` : patch size
