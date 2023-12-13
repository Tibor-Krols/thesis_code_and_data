import os
import pandas as pd
from utils.paths import *
from baseline_lpp.baseline_trainset_per_volume import run_baseline_volumes


def main():
    volumes_dict,gen_sentences = run_baseline_volumes()
    df = pd.DataFrame(
        {
            'sections':volumes_dict['section'],
            'volume_idx': volumes_dict['volume'],
            'prediction': gen_sentences,
            'ground_truth': volumes_dict['ground_truth'],
            'nr_words_per_volumes': volumes_dict['nwords']
        }
    )
    filename = f'baseline_predictions_per_volume_full_text.csv'
    filepath = pred_path/'baseline'/'embeddings'
    os.makedirs(filepath, exist_ok=True)
    df.to_csv(filepath/filename)


    # Split the strings in the 'predictions' column into lists of words
    df['prediction'] = df['prediction'].str.split()
    # Use explode to create a new row for each word
    df_expanded = df.explode('prediction')
    filename = f'baseline_word_predictions_per_volume_full_text.csv'
    filepath = pred_path/'baseline'/'embeddings'
    os.makedirs(filepath, exist_ok=True)
    df_expanded.to_csv(filepath/filename)
    print('done')
if __name__ == '__main__':
    main()
