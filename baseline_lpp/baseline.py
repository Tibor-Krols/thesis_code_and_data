from nltk.tokenize import word_tokenize, sent_tokenize
import random
import re
from collections import defaultdict
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


def generate_text(word_probabilities, n):
    """
     Generate the sequence of n words based on the probability distribution
    :param word_probabilities:
    :param n:
    :return:
    """
    sequence = []
    random.seed(42) #for reproducibility
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
    # preproc_sentences = preproc_sentences[:10] # TODO remove this, for now for saving compute
    gen_sentences = generate_probability_sequences(
        sentences=preproc_sentences,
        word_probabilities=word_probabilities
    )
    return preproc_sentences,gen_sentences



if __name__ == "__main__":
    preproc_sentences,gen_sentences = run_baseline()
    print('done')

