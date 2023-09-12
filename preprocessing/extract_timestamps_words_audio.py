from utils.paths  import *
import os
import re
from datetime import timedelta
import json


def load_textgrid(section):
    filename = f"lppEN_section{section}.TextGrid.txt"
    #TODO: change to TextGrid and not .txt?
    file_path = os.path.join(annot_path, 'EN', filename)
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


def load_audio_timestamps(section,language = 'EN'):
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
    wordlist = [{
        'word':w['word'],
        'start': timedelta(seconds=w['start']),
        'end':timedelta(seconds=w['end'])
    } for w in wordlist]
    return wordlist


def create_all_word_timestamp_files():
    # load textgrid file
    #TODO: change to 9 sections (range(1,10)
    for section in range(1,3):
        data = load_textgrid(section=section)
        wordlist = extract_word_timestamps(data)
        save_audio_timestamps(wordlist=wordlist,section=section)



if __name__ == '__main__':
    print('creating all files')
    create_all_word_timestamp_files()
