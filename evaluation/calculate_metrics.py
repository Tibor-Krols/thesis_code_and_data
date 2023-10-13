import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
from baseline_lpp.baseline import run_baseline
from utils.paths import *
from jiwer import wer
from bert_score import BERTScorer
import socket
from urllib3.connection import HTTPConnection
from utils import file_saving
"""
BERTScore (https://arxiv.org/abs/1904.09675)
"""
class BERTSCORE(object):
    """
    copied from https://github.com/HuthLab/semantic-decoding/blob/main/decoding/utils_eval.py
    """
    def __init__(self, idf_sents=None, rescale = True, score = "f"):
        self.metric = BERTScorer(lang = "en", rescale_with_baseline = rescale, idf = (idf_sents is not None), idf_sents = idf_sents)
        if score == "precision": self.score_id = 0
        elif score == "recall": self.score_id = 1
        else: self.score_id = 2

    def score(self, ref, pred):
        ref_strings = [" ".join(x) for x in ref]
        pred_strings = [" ".join(x) for x in pred]
        return self.metric.score(cands = pred_strings, refs = ref_strings,verbose=True)[self.score_id].numpy()


def create_metrics_df(ground_truth: list[str], predicted: list[str], include_bert_score = False) -> pd.DataFrame:
    """
    calculates metrics
    :param ground_truth:
    :param predicted:
    :return:
    """


    # Initialize ROUGE scorer
    ROUGE_SCORER = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    if include_bert_score:
        # set higher timeout mb to prevent connection timeout when downloading pytorch_model.bin
        HTTPConnection.default_socket_options = (
                HTTPConnection.default_socket_options + [
            (socket.SOL_SOCKET, socket.SO_SNDBUF, 2000000),
            (socket.SOL_SOCKET, socket.SO_RCVBUF, 2000000)
        ])
        BERT_SCORER = BERTSCORE()
    # Initialize variables to accumulate scores
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    bleu_scores = []
    meteor_scores = []
    wer_scores = []
    # bert_scores = []

    # Loop through each pair of ground truth and predicted sentences
    for gt, pred in zip(ground_truth, predicted):
        # Tokenize the sentences
        gt_tokens = gt.split()
        pred_tokens = pred.split()

        # Calculate ROUGE scores
        rouge_scores = ROUGE_SCORER.score(' '.join(gt_tokens), ' '.join(pred_tokens))
        rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
        rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
        rougeL_scores.append(rouge_scores['rougeL'].fmeasure)

        # Calculate BLEU score
        bleu_score = sentence_bleu(gt_tokens, pred_tokens)
        bleu_scores.append(bleu_score)

        # Calculate METEOR score
        meteor_score_value = single_meteor_score(gt_tokens, pred_tokens)
        meteor_scores.append(meteor_score_value)

        # Calculate word error rate (WER)
        wer_score = wer(reference=gt,hypothesis=pred)
        wer_scores.append(wer_score)

    # add bert score if specified
    bert_scores = None
    if include_bert_score:
        bert_scores = BERT_SCORER.score(ref=ground_truth, pred=predicted)
        # bert_scores.append(bs)

    df_scores = pd.DataFrame(
        {
            'gt': ground_truth,
            'pred': predicted,
            'rouge1': rouge1_scores,
            'rouge2': rouge2_scores,
            'rougeL': rougeL_scores,
            'bleu': bleu_scores,
            'meteor': meteor_scores,
            'wer':wer_scores,
            'bert_score': bert_scores

        }
    )
    return df_scores


def save_baseline_metrics(include_bert_score:bool=False):
    preproc_sentences_base, gen_sentences_base = run_baseline()
    df_metrics_baseline = create_metrics_df(ground_truth=preproc_sentences_base, predicted=gen_sentences_base, include_bert_score=include_bert_score)
    filename = 'baseline_metrics.csv'
    save_path = os.path.join(eval_path, 'metrics')
    file_path = os.path.join(save_path, filename)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df_metrics_baseline.to_csv(file_path, index=False)
    print(f'saved {file_path}')


def save_bayesian_volume_metrics_participant(filename,filepath,include_bert_score=False):
    if not filename.endswith('.csv'):
        filename +='.csv'
    df_pred = pd.read_csv(filepath/filename)
    save_filename = 'metrics_' + filename
    df_metrics_bayes = create_metrics_df(
        ground_truth=df_pred['ground_truth'],
        predicted=df_pred['pred_text'],
        include_bert_score=include_bert_score)
    savepath = eval_path/'metrics'/'bayesian'
    file_saving.save_df(df_metrics_bayes,save_filename,save_path=savepath)

if __name__ == "__main__":
    save_baseline_metrics(include_bert_score =False)
    print('done')
