from dataset_loader.dataset import LPPDataset
from preprocessing.audio.extract_timestamps_words_audio import extract_sentences
from training.train_test_split import train_test_split_lpp
from dataset_loader.section_participant_base import BaseSectionParticipant
from utils.paths import *
from torch import save,load
from tqdm import tqdm
from models.bayesian.calculate_fmri_averages import calculate_averages

# TODO: train test split sections randomly for each participant
def calculate_averages_participant(dataset,indices_participant_sections,participant):
    calculate_averages(dataset,indices_participant_sections=indices_participant_sections,participant=participant)

def load_averages_participant(participant:str,folderpath = data_path / 'word_averages_participants' ):
    """
    example filename 'word_averages_fmri_dict_EN-sub57.pt'
    :param participant:
    :param folderpath:
    :return:
    """
    filename = f'{participant}_word_averages_fmri_dict.pt'
    filepath = folderpath/filename
    return load(filepath)

def main():
    dataset = LPPDataset()
    participant = 'sub-EN057'
    participant_indices = dataset.get_participant_samples_indices(participant)
    train_indices = participant_indices[:8]
    test_indices = [participant_indices[-1]]

    # calculate_averages_participant(dataset=dataset,
    #                                indices_participant_sections=train_indices,
    #                                participant=participant)

    participant_dict = load_averages_participant(participant)

    print("done")

if __name__ == '__main__':
    main()