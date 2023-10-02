import nibabel as nib
import matplotlib.pyplot as plt
from utils.paths  import *

import sys
import os
import nibabel as nib
import numpy as np
# from scipy.stats import pearsonr
import os

# TODO: chang to path variables util
src_path = r"/"
data_path = os.path.join(src_path,r'dataset_sample\derivatives')
subjects_en = ["subj57"] # how they did it in LPP code
subjects_en = ['sub-EN057']

# TODO: loop over participants:
# TODO: loop over files per participant (1-9)
for subject_en in subjects_en:
    subject_path = os.path.join(data_path,f"{subject_en}")
    file_path = os.path.join(subject_path,r'func\run17.e01.nii')


subject_en = 'sub-EN057'
subject_path = os.path.join(data_path,f"{subject_en}")
# section_nrs = [8,9,10,11,12,13,14,15,16] #TODO: extract these automatically per participant
section_nrs = [8,9,10,11,12,13,14,15,16] #TODO: extract these automatically per participant
sections = []
for i in section_nrs:
    file_path = os.path.join(subject_path,'func',f'resampled.run{i:02d}.e0123_medn_afw.nii')
    print(file_path)
    subj_data = nib.load(file_path).get_fdata()
    sections.append(subj_data)
# mask = nib.load("lpp_%s%d_groupmask_thrs80.nii" %(lang,len(subjects))).get_fdata()

for i in range(len(sections)):
    print()


print('end')



# Specify the time point or volume you want to select (e.g., volume 0)
volume_index = 100

# Select the 3D brain image (volume) at the specified time point
selected_volume = subj_data[..., volume_index]

# img = subj_data[0] shows weird images
img = selected_volume
plt.style.use('default')
fig, axes = plt.subplots(5,5, figsize=(12,12))
for i, ax in enumerate(axes.reshape(-1)):
    ax.imshow(img[:,:,40 + i])
plt.show()


print('end')