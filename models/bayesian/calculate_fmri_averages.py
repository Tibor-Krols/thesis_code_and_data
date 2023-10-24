from dataset_loader.dataset import LPPDataset
from preprocessing.audio.extract_timestamps_words_audio import extract_sentences
from training.train_test_split import train_test_split_lpp
from dataset_loader.section_participant_base import BaseSectionParticipant
from utils.paths import *
from torch import save,load
from tqdm import tqdm


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


def load_averages(filepath = data_path / 'word_averages'/'word_averages_fmri_dict.pt' ):
    return load(filepath)

def main():
    dataset = LPPDataset()
    train_indices,test_indices = train_test_split_lpp(dataset)
    calculate_averages(dataset,indices_participant_sections=train_indices)
    avg_fmri_word_dict = load_averages()
    print('test')



if __name__ == '__main__':
    main()