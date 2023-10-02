import nibabel as nib
import matplotlib.pyplot as plt
from utils.paths  import *

import sys
import os
import nibabel as nib
import numpy as np
# from scipy.stats import pearsonr
import os
from utils.load_participant_section import get_section_filename_map_participant

# TODO: chang to path variables util
# src_path = r"/"
data_path = os.path.join(data_path,'derivatives')
subjects_en = ["subj57"] # how they did it in LPP code
subjects_en = ['sub-EN057']

# TODO: loop over participants:
# TODO: loop over files per participant (1-9)
# for subject_en in subjects_en:
#     subject_path = os.path.join(data_path,f"{subject_en}")
#     file_path = os.path.join(subject_path,r'func\run17.e01.nii')



# def load_fmri_participant_section(participant :str,section:int, sections_path_dict: dict):
#     file_path = sections_path_dict[section]
#     return nib.load(file_path).get_fdata()

# subject_en = 'sub-EN057'
participant = 'sub-EN057'
sections_path_dict = get_section_filename_map_participant(participant=participant)
# subject_path = os.path.join(data_path,f"{subject_en}")
section_nrs = range(1,10)
section_nrs = range(1,3)#smaller subset for testing

sections_fmri_dict = {}
for i in section_nrs:
    # file_path = os.path.join(subject_path,'func',f'resampled.run{i:02d}.e0123_medn_afw.nii')
    file_path = sections_path_dict[i]
    # file_path = os.path.join(subject_path,'func', f'sub-EN057_task-lppEN_run-15_space-MNIColin27_desc-preproc_bold.nii.gz')
    # file_path = os.path.join(subject_path,'func', f'resampled.run15.e0123_medn_afw.nii')

    # subj_data = nib.load(file_path).get_fdata()
    # sections.append(subj_data)
    sections_fmri_dict[i] = nib.load(file_path).get_fdata()
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