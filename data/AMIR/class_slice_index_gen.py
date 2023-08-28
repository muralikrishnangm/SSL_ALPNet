# ## Generate class-slice indexing table for experiments
# 
# 
# ### Overview
# 
# This is for experiment setting up for simulating few-shot image segmentation scenarios
# 
# Input: pre-processed images and their ground-truth labels
# 
# Output: a `json` file for class-slice indexing


# get_ipython().run_line_magic('reset', '')
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
import numpy as np
import os
import glob
import SimpleITK as sitk
import sys
import json
sys.path.insert(0, '../../dataloaders/')
import niftiio as nio

base_dir='./amir_MR_normalized/test_data_wLabel'
IMG_BNAME=f'{base_dir}/image_*.nii.gz'
SEG_BNAME=f'{base_dir}/label_*.nii.gz'

imgs = glob.glob(IMG_BNAME)
segs = glob.glob(SEG_BNAME)
imgs = [ fid for fid in sorted(imgs, key = lambda x: int(x.split("_")[-1].split(".nii.gz")[0])  ) ]
segs = [ fid for fid in sorted(segs, key = lambda x: int(x.split("_")[-1].split(".nii.gz")[0])  ) ]

print(f'images: {imgs}')

print(f'labels: {segs}')

classmap = {}
LABEL_NAME = ["BG", "PORES", "CEMPASTE", "DMICRO", "BMICRO"]     

MIN_TP = 1 # minimum number of positive label pixels to be recorded. Use >100 when training with manual annotations for more stable training

fid = f'{base_dir}/classmap_{MIN_TP}.json' # name of the output file. 
for _lb in LABEL_NAME:
    classmap[_lb] = {}
    for _sid in segs:
        pid = _sid.split("_")[-1].split(".nii.gz")[0]
        classmap[_lb][pid] = []
# print(f'classmap: {classmap}')

for seg in segs:
    pid = seg.split("_")[-1].split(".nii.gz")[0]
    lb_vol = nio.read_nii_bysitk(seg)
    n_slice = lb_vol.shape[0]
    for slc in range(n_slice):
        for cls in range(len(LABEL_NAME)):
            if cls in lb_vol[slc, ...]:  # this is where we need the labels of each image slice
                if np.sum( lb_vol[slc, ...]) >= MIN_TP:
                    classmap[LABEL_NAME[cls]][str(pid)].append(slc)
    print(f'pid {str(pid)} finished!')
# print(f'classmap: {classmap}')

with open(fid, 'w') as fopen:
    json.dump(classmap, fopen)
    fopen.close()  
    


