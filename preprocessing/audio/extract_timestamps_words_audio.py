from utils.paths  import *
import os
import re
from datetime import timedelta
import json

def load_textgrid(section):
    filename = f"lppEN_section{section}.TextGrid"
    file_path = os.path.join(annot_path, 'EN', filename)
    if section in [3,4,5,6]:
        # little hack because the encoding of these sections is different for some reason
        with open(file_path, 'r',encoding='utf-16') as f:
            data = f.read()
    else:
        with open(file_path, 'r') as f:
            data = f.read()
    return data
def extract_word_timestamps(data:str)-> list[dict]:
    """ 
    strongly inspired by https://stackoverflow.com/questions/41702154/how-do-i-read-the-variables-of-textgrid-file-into-python
    """
    data = data.split('\n')
    word_list =[]
    for line in data[9:]:  #informations needed begin on the 9th lines
      # line = re.sub('\n','',line) #as there's \n at the end of every sentence.
      line = re.sub(' ','',line) #as there's \n at the end of every sentence.
      line = re.sub ('^ *','',line) #To remove any special characters
      linepair = line.split('=')
      if len(linepair) == 2:
        if linepair[0] == 'xmin':
            xmin = linepair[1]
        if linepair[0] == 'xmax':
            xmax = linepair[1]
        if linepair[0] == 'text':
            word = linepair[1]
            # word = re.sub("'",'',word)
            word = re.sub('"','',word)
            word_list.append({
                'word':str(word),
                'start':float(xmin),
                'end':float(xmax)
                })
    return word_list

def save_audio_timestamps(wordlist,section,language = 'EN'):
    filename = f'audio_word_timestamps{language}_section{section}.txt'
    save_path = os.path.join(annot_path,'EN','processed')
    file_path = os.path.join(save_path, filename)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(file_path, 'w') as fout:
        json.dump(wordlist, fout)


def load_audio_timestamps(section,language = 'EN',as_timedelta = True):
    """
    loads the json file and converts time in seconds to timedelta object
    :param section:
    :param language:
    :return:
    """
    filename = f'audio_word_timestamps{language}_section{section}.txt'
    save_path = os.path.join(annot_path,'EN','processed')
    file_path = os.path.join(save_path, filename)
    # Open and read the JSON file
    with open(file_path, 'r') as json_file:
        wordlist = json.load(json_file)
    if as_timedelta:
        wordlist = [{
            'word':w['word'],
            'start': timedelta(seconds=w['start']),
            'end':timedelta(seconds=w['end'])
        } for w in wordlist]
    else:
        wordlist = [{
            'word': w['word'],
            'start': w['start'],
            'end': w['end']
        } for w in wordlist]
    return wordlist


def create_all_word_timestamp_files():
    # load textgrid file
    #TODO: change to 9 sections (range(1,10)
    for section in range(1,10):
    # for section in range(3,7):
        data = load_textgrid(section=section)
        wordlist = extract_word_timestamps(data)
        save_audio_timestamps(wordlist=wordlist,section=section)


def extract_sentences(wordlist):
    """
    splits sentences based on # split
    :param wordlist: a list of words where each word is a dict with word, start and end as keys
    {'word': 'once', 'start': datetime.timedelta(microseconds=113200), 'end': datetime.timedelta(microseconds=728200)}
    :return: a list of sentences. where a sentence is a list of words dicts
    """
    sentences = []
    sentence = []

    for entry in wordlist:
        word = entry['word']

        if word == '#':
            if sentence:
                sentences.append(sentence)
            sentence = []
        else:
            sentence.append(entry)

    # Add the last sentence if it exists
    if sentence:
        sentences.append(sentence)

    return sentences


# TODO: make load_section_timestamps and map for specific section and map_words_to_volumes to work for one section
def load_section_timestamps(section:int, as_timedelta = True):
    section_dict = {
        section: load_audio_timestamps(section=section,as_timedelta=as_timedelta)
    }
    return section_dict
def load_all_sections_timestamps(as_timedelta = True):
    section_dict = {}
    for section in range(1, 10):
        wordslist = load_audio_timestamps(section=section,as_timedelta=as_timedelta)
        section_dict[section] = wordslist
    return section_dict

def load_full_book()-> str:
    book_text = []
    for section in range(1, 10):
        section_text = load_audio_timestamps(section)
        section_text = [w['word'] for w in section_text if w['word'] != '#']
        book_text += section_text
    book_text = ' '.join(section for section in book_text)
    return book_text
def load_full_book_sections():
    sections_text = {}
    for section in range(1, 10):
        section_text = load_audio_timestamps(section)
        section_text = [w['word'] for w in section_text if w['word'] != '#']
        sections_text[section]= " ".join(section_text)
        # sections_text += [" ".join(section_text)]
    # sections_text = {i+1:text for i,text in enumerate(sections_text)}
    # book_text = ' '.join(section for section in book_text)
    return sections_text

if __name__ == '__main__':
    print('creating all files')
    create_all_word_timestamp_files()



# word_average_dict = {
#     'w1':v1,
#     'w2':v2,
#     ...
#     'wn':vn
# }