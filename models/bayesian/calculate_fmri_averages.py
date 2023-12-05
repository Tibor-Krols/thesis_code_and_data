from dataset_loader.dataset import LPPDataset
from preprocessing.audio.extract_timestamps_words_audio import extract_sentences
from training.train_test_split import train_test_split_lpp
from dataset_loader.section_participant_base import BaseSectionParticipant
from utils.paths import *
from torch import save,load
from tqdm import tqdm
import torch
import nilearn

def calculate_averages(dataset,indices_participant_sections,participant = ""):
    sum_fmri_word_dict = {}
    count_fmri_word_dict = {}

    # indices_participant_sections = indices_participant_sections[:18]
    # make sum and count per word
    for i in tqdm(indices_participant_sections):
        ps = BaseSectionParticipant(dataset[i],include_volume_words_dict =True)
        for vol_idx,words in ps.volume_words_dict.items():
            for word in words:
                # initialize word count and sum
                if word not in sum_fmri_word_dict.keys():
                    sum_fmri_word_dict[word] = ps[vol_idx]
                    count_fmri_word_dict[word] = 1
                else:
                    sum_fmri_word_dict[word] += ps[vol_idx]
                    count_fmri_word_dict[word] += 1

    # calculate_averages
    avg_fmri_word_dict = {}
    for word,count_words in count_fmri_word_dict.items():
        avg_fmri_word_dict[word] = sum_fmri_word_dict[word]/count_words

    # save dict
    if participant == "":
        save_path = data_path / 'word_averages'
        os.makedirs(save_path,exist_ok=True)
        file_path = save_path / f'word_averages_fmri_dict.pt'
        save(avg_fmri_word_dict, file_path)
    else:
        save_path = data_path / 'word_averages_participants'
        os.makedirs(save_path,exist_ok=True)
        file_path = save_path / f'{participant}_word_averages_fmri_dict.pt'
        save(avg_fmri_word_dict, file_path)
def calculate_std(dataset,indices_participant_sections,cortical_region:str,participant = ""):
    # to avoid circular imports
    from models.bayesian.fmri_averages_per_participant import load_averages_participant
    from utils.cortical_masking import make_nifti_image_from_tensor, get_oxford_mask

    sum_val_minus_mean_word_dict = {}
    count_fmri_word_dict = {}

    # load averages dict for specific mask
    if participant == "":
        avg_fmri_word_dict = load_averages()
    else:
        avg_fmri_word_dict = load_averages_participant(participant,cortical_regions = [cortical_region])

    # load cortical mask
    # cortical_mask =
    cortical_mask = get_oxford_mask(cortical_regions= [cortical_region])

    # indices_participant_sections = indices_participant_sections[:18]
    # make sum and count per word
    for i in tqdm(indices_participant_sections):
        ps = BaseSectionParticipant(dataset[i],include_volume_words_dict =True)
        for vol_idx,words in ps.volume_words_dict.items():
            if cortical_mask is not None:
                vol = torch.tensor(nilearn.masking.apply_mask(
                    make_nifti_image_from_tensor(ps[vol_idx]),
                    mask_img=cortical_mask))
            else:
                vol = ps[vol_idx]
            for word in words:
                # initialize word count and sum
                if word not in sum_val_minus_mean_word_dict.keys():
                    sum_val_minus_mean_word_dict[word] = (vol - avg_fmri_word_dict[word])**2
                    count_fmri_word_dict[word] = 1
                else:
                    sum_val_minus_mean_word_dict[word] += (vol - avg_fmri_word_dict[word])**2
                    count_fmri_word_dict[word] += 1

    # calculate standard deviation (std)
    std_fmri_word_dict = {}
    for word,count_words in count_fmri_word_dict.items():
        std_fmri_word_dict[word] = torch.sqrt(sum_val_minus_mean_word_dict[word]/count_words)

    # save dict
    if participant == "":
        save_path = data_path / 'word_std'
        os.makedirs(save_path,exist_ok=True)
        file_path = save_path / f'word_std_fmri_dict.pt'
        save(std_fmri_word_dict, file_path)
    else:
        save_path = data_path / 'word_std_participants'
        os.makedirs(save_path,exist_ok=True)
        file_path = save_path / f'{participant}_word_std_fmri_dict.pt'
        save(std_fmri_word_dict, file_path)

def load_averages(filepath = data_path / 'word_averages'/'word_averages_fmri_dict.pt' ):
    return load(filepath)

def main():
    dataset = LPPDataset()
    train_indices,test_indices = train_test_split_lpp(dataset)
    calculate_std(dataset,
                  indices_participant_sections=train_indices,
                  cortical_region='Superior Temporal Gyrus, anterior division',
                  participant='sub-EN057'
                  )
    calculate_averages(dataset,indices_participant_sections=train_indices)
    avg_fmri_word_dict = load_averages()
    print('test')



if __name__ == '__main__':
    main()