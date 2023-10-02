import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
import random
import re
from collections import defaultdict
from utils.paths import *
import numpy as np

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

def get_word_probabilities(text):
    # Step 1: Preprocess text
    preprocessed_text = preprocess_text(text)
    # Step 2: Tokenize the text into words
    words = preprocessed_text.split()
    # Step 3: Build a probability distribution based on word frequencies
    word_frequency = defaultdict(int)
    for word in words:
        word_frequency[word] += 1
    total_words = len(words)
    word_probabilities = {word: count / total_words for word, count in word_frequency.items()}
    return word_probabilities


def beam_search( seed_text, beam_width, max_length,word_probabilities):
    # Initialize the list of candidates with the seed text
    candidates = [(seed_text, 1.0)]
    prob_dist = list(word_probabilities.values())
    vocab = list(word_probabilities.keys())
    for _ in range(max_length):
        new_candidates = []

        for candidate, score in candidates:
            # Get the probability distribution over the vocabulary for the next word
            # prob_dist = model.get_word_probabilities(candidate)
            # TODO: get prob_dist from language model, map to lpp vocab and multiply prob_dist with posterior prob dist of brainscan
            # Select the top 'beam_width' words
            top_words_indices = np.argsort(prob_dist)[-beam_width:]

            # random_word = random.choices(list(word_probabilities.keys()),
            #                              list(word_probabilities.values()))[0]
            for word_index in top_words_indices:
                # next_word = model.vocab[word_index]
                next_word =  vocab[word_index]
                new_candidate = candidate + " " + next_word
                new_score = score * prob_dist[word_index]

                new_candidates.append((new_candidate, new_score))

        # Sort and select the top 'beam_width' candidates
        new_candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = new_candidates[:beam_width]

    # Select the candidate with the highest score as the generated sentence
    best_candidate, _ = max(candidates, key=lambda x: x[1])
    return best_candidate





def generate_text(word_probabilities, n):
    """
     Generate the sequence of n words based on the probability distribution
    :param word_probabilities:
    :param n:
    :return:
    """
    sequence = []
    # random.seed(42) #for reproducibility
    for _ in range(n):
        random_word = random.choices(list(word_probabilities.keys()),
                                     list(word_probabilities.values()))[0]
        sequence.append(random_word)
    return ' '.join(sequence)

def generate_probability_sequences(sentences,word_probabilities):
    gen_sentences = []
    for s in sentences:
        nwords = len(s.split())
        gen_sent = generate_text(word_probabilities=word_probabilities,n=nwords)
        gen_sentences.append(gen_sent)
    return gen_sentences


def run_baseline_beam_search():
    # Example usage:
    seed_text = "Once upon a time"
    beam_width = 3
    max_length = 20
    # Read and preprocess the text file
    text = load_lpp_book()
    preprocessed_text = preprocess_text(text=text)

    # calculate word probabilities
    word_probabilities = get_word_probabilities(preprocessed_text)
    generated_sentence = beam_search( seed_text, beam_width, max_length, word_probabilities)
    print(generated_sentence)


def run_baseline():
    """
    runs baseline
    :return: list of ground truth sentences, list of predicted sentences based on probability
    """
    # Read and preprocess the text file
    text = load_lpp_book()
    preprocessed_text = preprocess_text(text=text)

    # calculate word probabilities
    word_probabilities = get_word_probabilities(preprocessed_text)

    # extract sentences and preprocess
    sentences = sent_tokenize(text)
    preproc_sentences = [preprocess_text(s) for s in sentences]

    # generate probability sentences with the same lenght
    # preproc_sentences = preproc_sentences[:5] # TODO remove this, for now for saving compute
    gen_sentences = generate_probability_sequences(
        sentences=preproc_sentences,
        word_probabilities=word_probabilities
    )
    return preproc_sentences,gen_sentences

def save_baseline_preds(sentences,gen_sentences):
    df = pd.DataFrame(
        {
            'sentences':sentences,
            'predictions':gen_sentences
        }
    )

    filename = 'baseline_predictions.csv'
    save_path = os.path.join(eval_path, 'predictions')
    file_path = os.path.join(save_path, filename)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df.to_csv(file_path, index=False)


if __name__ == "__main__":
    run_baseline_beam_search()
    preproc_sentences,gen_sentences = run_baseline()
    save_baseline_preds(sentences=preproc_sentences,gen_sentences=gen_sentences)
    print('done')

