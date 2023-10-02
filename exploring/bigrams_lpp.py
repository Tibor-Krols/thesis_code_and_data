import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
import random
import re
from collections import defaultdict
from utils.paths import *
import nltk
from baseline_lpp.baseline import preprocess_text,load_lpp_book


text = load_lpp_book()
preprocessed_text = preprocess_text(text=text)



tokens = nltk.word_tokenize(preprocessed_text)
bigrams = nltk.bigrams(tokens)
frequencybi = nltk.FreqDist(bigrams)

trigrams = nltk.trigrams(tokens)
frequencytri = nltk.FreqDist(trigrams)


print(len(preprocessed_text))