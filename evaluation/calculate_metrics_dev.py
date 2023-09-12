import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
from baseline_lpp.baseline import run_baseline
from utils.paths import *
from jiwer import wer

def create_metrics_df(ground_truth: list[str], predicted: list[str]) -> pd.DataFrame:
    """
    calculates metrics for a
    :param ground_truth:
    :param predicted:
    :return:
    """
    # Initialize ROUGE scorer
    ROUGE_SCORER = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Initialize variables to accumulate scores
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    bleu_scores = []
    meteor_scores = []
    wer_scores = []
    bert_scores = []

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
        # print(pred_tokens)
        bleu_score = sentence_bleu(gt_tokens, pred_tokens)
        # bleu_score = corpus_bleu([gt_tokens],pred_tokens)
        bleu_scores.append(bleu_score)

        # Calculate METEOR score
        meteor_score_value = single_meteor_score(gt_tokens, pred_tokens)
        meteor_scores.append(meteor_score_value)

        # Calculate word error rate (WER)
        wer_score = wer(reference=gt,hypothesis=pred)
        wer_scores.append(wer_score)

        # TODO: add BERTScore

    df_scores = pd.DataFrame(
        {
            'gt': ground_truth,
            'pred': predicted,
            'rouge1': rouge1_scores,
            'rouge2': rouge2_scores,
            'rougeL': rougeL_scores,
            'bleu': bleu_scores,
            'meteor': meteor_scores,
            'wer':wer_scores

        }
    )
    return df_scores


def save_baseline_metrics():
    preproc_sentences_base, gen_sentences_base = run_baseline()
    df_metrics_baseline = create_metrics_df(ground_truth=preproc_sentences_base, predicted=gen_sentences_base)
    filename = 'baseline_metrics.csv'
    save_path = os.path.join(eval_path, 'metrics')
    file_path = os.path.join(save_path, filename)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df_metrics_baseline.to_csv(file_path, index=False)
    print(f'saved {file_path}')

if __name__ == "__main__":
    save_baseline_metrics()
# # Calculate average scores
# avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
# avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
# avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)
# avg_bleu = sum(bleu_scores) / len(bleu_scores)
# avg_meteor = sum(meteor_scores) / len(meteor_scores)
#
# # Add the average scores to the DataFrame
# metrics_df.loc[0] = [avg_rouge1, avg_rouge2, avg_rougeL, avg_bleu, avg_meteor]
#
# # Display the DataFrame


print('done')
