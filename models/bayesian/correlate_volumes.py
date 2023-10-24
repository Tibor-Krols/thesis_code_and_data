from models.bayesian.calculate_fmri_averages import load_averages
from dataset_loader.dataset import LPPDataset
from dataset_loader.section_participant_base import BaseSectionParticipant
import torch
import torch.nn.functional as F
from preprocessing.audio.extract_vocabulary import Vocab
def calulate_correlation(vol1,vol2):
    # TODO: parallelize correlation by stacking average fmri volumes and making sure the top one is the to be predicted volume.
    # x = torch.cat((vol1.reshape(-1),vol2.reshape(-1)), dim=0)
    # x2 = torch.tensor([vol1.reshape(-1),vol2.reshape(-1)])
    x = torch.vstack((vol1.reshape(-1),vol2.reshape(-1)))
    # Stack the two row vectors into a 2D tensor (matrix)
    # stacked_matrix = torch.cat((row_vector1.unsqueeze(0), row_vector2.unsqueeze(0)), dim=0)

    # print(stacked_matrix)
    correlation_coefficient = torch.corrcoef(x)
    return correlation_coefficient

def calulate_word_correlations(avg_fmri_word_dict, volume):
    cors = torch.stack([calulate_correlation(volume,vol_w)[0,1] for vol_w in avg_fmri_word_dict.values() ])
    return cors


def calculate_mse(vol1,vol2):
    mse = torch.sqrt(torch.mean((vol1.reshape(-1)-vol2.reshape(-1)) ** 2))
    return mse

def calculate_word_mse(avg_fmri_word_dict, volume):
    return torch.stack([calculate_mse(volume,vol_w) for vol_w in avg_fmri_word_dict.values() ])


def stack_mean_word_vocab(avg_fmri_word_dict):
    stack = torch.stack([vol.reshape(-1) for vol in avg_fmri_word_dict.values()])
    # for word,mean_fmri_word in avg_fmri_word_dict.items():

def normalize_correlation_vector_vocab(cor_vec):
    norm_vec = (cor_vec - cor_vec.mean()) / cor_vec.std()
    return norm_vec

def _softmax(vec):
    return F.softmax(vec, dim=0)

def softmax_dict(word_dict:dict)-> dict:
    # Extract the values from the dictionary
    values = torch.tensor(list(word_dict.values()))
    # Apply softmax to the tensor
    softmax_values = F.softmax(values, dim=0)
    # Create a new dictionary with the softmax values
    softmax_dict = {word: softmax.item() for word, softmax in zip(word_dict.keys(), softmax_values)}
    return softmax_dict

def map_corr_to_vocab(corr_vec, words_map, vocab = Vocab(),missing_vocab_value = 0):
    # make a dict from word list and correlation vector
    dict_vol = dict([(word,val) for word,val in zip(words_map,corr_vec)])

    # assing default correlation value to out of vocabulary words
    # dict_oov = dict((word,missing_vocab_value) for word in vocab.dataset_vocab if word not in words_map)
    # return dict_vol | dict_oov #returns combination of both dict
    return dict_vol


def main():
    dataset = LPPDataset()
    ps = BaseSectionParticipant(dataset[0], include_volume_words_dict=True)
    avg_fmri_word_dict = load_averages()

    # vol1 = ps[1]
    # vol2 = avg_fmri_word_dict['magnificent']
    # corr = calulate_correlation(vol1, vol2)
    cor_vec_vol = calulate_word_correlations(avg_fmri_word_dict, vol1)
    norm_vec = normalize_correlation_vector_vocab(cor_vec_vol)
    softmax_vec = _softmax(cor_vec_vol)
    # index = list(avg_fmri_word_dict.keys()).index('magnificent')
    dict_vol = map_corr_to_vocab(
        corr_vec = norm_vec,
        words_map=list(avg_fmri_word_dict.keys())
    )


    print('end')

if __name__ == '__main__':
    main()