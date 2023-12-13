import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
import random
import re
from collections import defaultdict
from dataset_loader.dataset import LPPDataset
from dataset_loader.section_participant_base import BaseSectionParticipant

from preprocessing.audio.extract_timestamps_words_audio import load_full_book_sections
from training.train_test_split import get_train_test_sections
from utils.paths import *

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub('\n','',text)# Remove non-alphabetic characters
    return text

def load_lpp_book():
    path_folder = os.path.join(path, 'text')
    file_path = os.path.join(path_folder, 'the_little_prince.txt')
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def get_word_probabilities(text,preprocess = True):
    # Step 1: Preprocess text
    if preprocess:
        preprocessed_text = preprocess_text(text)
    else:
        preprocessed_text = text
    # Step 2: Tokenize the text into words
    words = preprocessed_text.split()
    # Step 3: Build a probability distribution based on word frequencies
    word_frequency = defaultdict(int)
    for word in words:
        word_frequency[word] += 1
    total_words = len(words)
    word_probabilities = {word: count / total_words for word, count in word_frequency.items()}
    return word_probabilities


def generate_text(word_probabilities, n,fixed_random_state=False):
    """
     Generate the sequence of n words based on the probability distribution
    :param word_probabilities:
    :param n:
    :return:
    """
    sequence = []
    if fixed_random_state:
        random.seed(42) #for reproducibility. dont use for baseline, as it only predicts the same words
    for _ in range(n):
        random_word = random.choices(list(word_probabilities.keys()),
                                     list(word_probabilities.values()))[0]
        sequence.append(random_word)
    return ' '.join(sequence)
def generate_text_most_probable(word_probabilities, n):
    """
     Generate the sequence of n words based on the probability distribution
    :param word_probabilities:
    :param n:
    :return:
    """
    # Get the n most probable words
    most_probable_words = sorted(word_probabilities, key=lambda word: word_probabilities[word], reverse=True)[:n]
    return ' '.join(most_probable_words)

def get_volumes_dict():
    dataset = LPPDataset()
    participant = dataset.participants[0] #just take the first participant
    sections = range(1, 10)
    participants_list = []
    sections_list = []
    volumes_list = []
    nwords_list = []
    ground_truth_list = []
    for section in sections:
        ps_idx = dataset.get_participant_section_index(participant,section)
        ps = BaseSectionParticipant(dataset[ps_idx], include_volume_words_dict=True)
        for vol_idx in range(ps.nr_fmri_frames):
            ground_truth_vol = ps.get_words_volume_idx(vol_idx)
            nwords_volume = len(ground_truth_vol)
            # add values to list
            participants_list.append(participant)
            sections_list.append(section)
            volumes_list.append(vol_idx)
            nwords_list.append(nwords_volume)
            ground_truth_list.append(' '.join(ground_truth_vol))

    nword_dict = {
        'participant':participants_list,
        'section':sections_list,
        'volume': volumes_list,
        'nwords': nwords_list,
        'ground_truth':ground_truth_list
    }
    return nword_dict

def generate_probability_sequences(volumes_dict,word_probabilities):
    gen_volumes_text = []
    for nwords in volumes_dict['nwords']:
        gen_sent = generate_text(word_probabilities=word_probabilities,n=nwords)
        gen_volumes_text.append(gen_sent)
    return gen_volumes_text

def load_train_set_text():
    """
    returns text that only from the train sections.
    :return:
    """
    train_sections, test_sections = get_train_test_sections()
    train_sections.sort()
    sections_text = load_full_book_sections()
    train_sections_text = {
        sect:sections_text[sect] for sect in sections_text.keys()
        if sect in train_sections
    }
    text = " ".join([t for t in train_sections_text.values()])
    return text

def run_baseline_volumes():
    """
    runs baseline
    :return: list of ground truth sentences, list of predicted sentences based on probability
    """
    # Read and preprocess the text file
    # text = load_train_set_text()
    # text = load_lpp_book()
    text_sections = load_full_book_sections()
    sections = list(text_sections.keys())
    text = ''
    for section in sections:
        text += text_sections[section]
        text += ' '

    # text = load_lpp_book()
    preprocessed_text = preprocess_text(text=text)

    # calculate word probabilities
    word_probabilities = get_word_probabilities(preprocessed_text)

    # extract sentences and preprocess
    # sentences = sent_tokenize(text)
    # preproc_sentences = [preprocess_text(s) for s in sentences]

    # generate probability sentences with the same lenght
    # preproc_sentences = preproc_sentences[:5] # TODO remove this, for now for saving compute
    volumes_dict = get_volumes_dict()

    gen_sentences = generate_probability_sequences(
        volumes_dict=volumes_dict,
        word_probabilities=word_probabilities
    )
    return volumes_dict,gen_sentences


def run_baseline_trainset_volumes():
    """
    runs baseline
    :return: list of ground truth sentences, list of predicted sentences based on probability
    """
    # Read and preprocess the text file
    text = load_train_set_text()
    # text = load_lpp_book()
    preprocessed_text = preprocess_text(text=text)

    # calculate word probabilities
    word_probabilities = get_word_probabilities(preprocessed_text)

    # extract sentences and preprocess
    # sentences = sent_tokenize(text)
    # preproc_sentences = [preprocess_text(s) for s in sentences]

    # generate probability sentences with the same lenght
    # preproc_sentences = preproc_sentences[:5] # TODO remove this, for now for saving compute
    volumes_dict = get_volumes_dict()

    gen_sentences = generate_probability_sequences(
        volumes_dict=volumes_dict,
        word_probabilities=word_probabilities
    )
    return volumes_dict['ground_truth'],gen_sentences

def save_baseline_preds(sentences,gen_sentences):
    df = pd.DataFrame(
        {
            'ground_truth':sentences,
            'pred_text':gen_sentences
        }
    )

    filename = 'baseline_trainset_per_volume_predictions.csv'
    save_path = pred_path /'baseline'
    file_path = os.path.join(save_path, filename)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df.to_csv(file_path, index=False)


if __name__ == "__main__":
    gt_volumes,gen_volumes = run_baseline_trainset_volumes()
    save_baseline_preds(sentences=gt_volumes,gen_sentences=gen_volumes)
    print('done')

