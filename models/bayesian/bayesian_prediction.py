from baseline_lpp.baseline import generate_text
from dataset_loader.dataset import LPPDataset
from dataset_loader.section_participant_base import BaseSectionParticipant
from models.bayesian.calculate_fmri_averages import load_averages
from models.bayesian.correlate_volumes import calulate_word_correlations, normalize_correlation_vector_vocab, \
    map_corr_to_vocab, softmax_dict
from baseline_lpp import baseline
from preprocessing.audio.extract_timestamps_words_audio import load_full_book
import torch

# from models.bayesian.correlate_volumes import

def get_likelihood_volume_ps_volume_dict(avg_fmri_word_dict,ps:BaseSectionParticipant,vol_idx:int):
    vol = ps[vol_idx]
    # calculate correlations
    cor_vec_vol = calulate_word_correlations(avg_fmri_word_dict, vol)
    # normalize correlations?
    # norm_vec = normalize_correlation_vector_vocab(cor_vec_vol)


    # map vector to vocab
    likelihood_dict_ps_volume = map_corr_to_vocab(
        corr_vec = cor_vec_vol,
        words_map=list(avg_fmri_word_dict.keys())
    )
    # remove empty string:
    likelihood_dict_ps_volume.pop("", None)
    # TODO: already softmax here??
    likelihood_dict_ps_volume = softmax_dict(likelihood_dict_ps_volume)
    return likelihood_dict_ps_volume


def get_prior_dict()-> dict[str:int]:
    book_text = load_full_book()
    prior_dict = baseline.get_word_probabilities(book_text,preprocess=False)

    # normalize or softmax?
    return prior_dict



def calculate_posterior_ps_volume(prior_dict, likelihood_dict):
    assert len(prior_dict) == len(likelihood_dict)
    vocab = list(prior_dict.keys())
    prior_values = torch.tensor(list(prior_dict.values()))
    likelihood_values = torch.tensor(list(likelihood_dict.values()))
    posterior_values = prior_values * likelihood_values
    posterior_dict = {w:v.item() for w,v in zip(vocab,posterior_values)}
    return posterior_dict

def predict_words_posterior(posterior_dict, nwords:int)->str:
    pred_words = generate_text(
        word_probabilities=posterior_dict,
        n=nwords
    )
    return pred_words


def predict_words_ps_volume(avg_fmri_word_dict,ps,vol_idx,prior_dict)-> tuple[list[str],list[str]]:
    """"
    returns a tuple of ground truth of volume, prediction of volume
    """
    # likelihood
    likelihood_dict_ps_volume = get_likelihood_volume_ps_volume_dict(
        avg_fmri_word_dict=avg_fmri_word_dict,
        ps=ps,
        vol_idx=vol_idx
    )

    # posterior
    posterior_dict = calculate_posterior_ps_volume(
        prior_dict=prior_dict,
        likelihood_dict=likelihood_dict_ps_volume
    )
    # ground_truth_volume
    ground_truth_vol = ps.get_words_volume_idx(vol_idx)
    nwords_volume = len(ground_truth_vol)
    # word generation
    pred_words_vol = predict_words_posterior(
        posterior_dict=posterior_dict,
        nwords=nwords_volume
    )
    return ground_truth_vol,pred_words_vol.split()
def main():
    dataset = LPPDataset()
    ps = BaseSectionParticipant(dataset[0], include_volume_words_dict=True)
    avg_fmri_word_dict = load_averages()


    vol_idx = 1

    #prior
    prior_dict = get_prior_dict()

    gt_vol, pred_vol = predict_words_ps_volume(
        avg_fmri_word_dict,
        ps,
        vol_idx,
        prior_dict
    )
    print(gt_vol,pred_vol)
    print('done')



if __name__ == '__main__':
    main()