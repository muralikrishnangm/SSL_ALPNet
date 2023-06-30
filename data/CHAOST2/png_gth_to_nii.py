#!/usr/bin/env python
# coding: utf-8

# ## Converting labels from png to nii file
# 
# 
# ### Overview
# 
# This is the first step for data preparation
# 
# Input: ground truth labels in `.png` format
# 
# Output: labels in `.nii` format, indexed by patient id

# In[13]:


# get_ipython().run_line_magic('reset', '')
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
import os
import glob

import numpy as np
import PIL
import matplotlib.pyplot as plt
import SimpleITK as sitk
import sys
sys.path.insert(0, '../../dataloaders/')
import niftiio as nio


# In[ ]:





# In[14]:


example = "./MR/1/T2SPIR/Ground/IMG-0002-00001.png" # example of ground-truth file name. 


# In[15]:


### search for scan ids
ids = os.listdir("./MR/")
OUT_DIR = './niis/T2SPIR/'


# In[16]:


ids


# In[17]:


#### Write them to nii files for the ease of loading in future
for curr_id in ids:
    pngs = glob.glob(f'./MR/{curr_id}/T2SPIR/Ground/*.png')
    pngs = sorted(pngs, key = lambda x: int(os.path.basename(x).split("-")[-1].split(".png")[0]))
    buffer = []

    for fid in pngs:
        buffer.append(PIL.Image.open(fid))

    vol = np.stack(buffer, axis = 0)
    # flip correction
    vol = np.flip(vol, axis = 1).copy()
    # remap values
    for new_val, old_val in enumerate(sorted(np.unique(vol))):
        vol[vol == old_val] = new_val

    # get reference    
    ref_img = f'./niis/T2SPIR/image_{curr_id}.nii.gz'
    img_o = sitk.ReadImage(ref_img)
    vol_o = nio.np2itk(img=vol, ref_obj=img_o)
    sitk.WriteImage(vol_o, f'{OUT_DIR}/label_{curr_id}.nii.gz')
    print(f'image with id {curr_id} has been saved!')

    


# In[ ]:




