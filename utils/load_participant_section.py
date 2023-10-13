import nibabel as nib
import matplotlib.pyplot as plt
from utils.paths  import *

import sys
import nibabel as nib
import numpy as np
# from scipy.stats import pearsonr
import os
from preprocessing.audio.extract_timestamps_words_audio import load_audio_timestamps, extract_sentences,load_all_sections_timestamps,load_section_timestamps


def get_section_filename_map_participant(participant : str):
    """
    dict that maps filenames and sections given a participant
    :param participant:
    :return: dict where key is section and value is path to the section file for this participant
    """
    subject_path = os.path.join(data_path,'derivatives', f"{participant}",'func')
    # List all the files in the directory and sort them by filename
    files = os.listdir(subject_path)
    files.sort()
    # files = [f for f in files if f.endswith('.nii')] #if using unzipped files
    files = [f for f in files if f.endswith('.gz')] #if using unzipped files

    # Initialize a dictionary to store the mapping
    file_dict = {}
    # map the files to the section nrs
    for i in range(len(files)):
        file_dict[i+1] = os.path.join(subject_path,files[i])
    return file_dict




# Function to find subdirectories containing 'EN' in their names
def find_en_directories(directory,lang = "EN"):
    en_directories = []
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            if lang in dir_name:
                en_directories.append(dir_name)
    return en_directories


def get_all_participants_fmri_paths():
    participants = find_en_directories(deriv_path)
    participants_dict = {}
    for p in participants:
        sections_path_dict = get_section_filename_map_participant(participant=p)
        participants_dict[p] = sections_path_dict
    return participants_dict


def load_participant_section_fmri(participant,section:int,participants_path_dict=None):
    if participants_path_dict is None:
        participants_path_dict = get_all_participants_fmri_paths()
        print(f"getting participant_path_dict...")
    file_path = participants_path_dict[participant][section]
    # file_path = sections_path_dict[i]
    return nib.load(file_path).get_fdata()


def map_words_to_volumes(sections_timestamps_dict, start_bound:float = .1):
    """
    takes as input the sections timestamp dict on word level and adds the indexes of the fmri volumes to the word dicts
    :param sections_timestamps_dict:
    :return:
    """
    new_sections_timestamps_dict = {}
    for section,section_words in sections_timestamps_dict.items():
        # select volumes of start and end timestamp of word
        volume_list = [(int((w['start']+start_bound)//2),int(w['end']//2)) for w in section_words]
        # convert start and end time to range of volumes
        volume_list = [list(range(start,end +1)) for start,end in volume_list]

        # add volumes to sections_timestamps_dict
        list_of_dicts_with_new_key = [{**d, 'volume_idx': volume} for d, volume in zip(section_words, volume_list)]
        new_sections_timestamps_dict[section] = list_of_dicts_with_new_key

        # result = [dict(item, elem='value') for item in myList]

    return new_sections_timestamps_dict


def main():
    participants_path_dict = get_all_participants_fmri_paths()
    sections_timestamps_dict = load_all_sections_timestamps(as_timedelta=False)
    sections_timestamps_dict = map_words_to_volumes(sections_timestamps_dict)

    participant = list(participants_path_dict.keys())[0]
    section = 1
    data = load_participant_section_fmri(
        participant=participant,
        section=section,
        participants_path_dict=participants_path_dict
    )

    print('test')

if __name__ == '__main__':
    main()