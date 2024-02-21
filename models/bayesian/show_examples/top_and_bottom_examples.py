from pathlib import Path

import numpy as np
import pandas as pd

from models.bayesian.per_participant_cosine_similarity import add_cosine_to_bayesian_df
from sklearn.metrics.pairwise import cosine_similarity

from models.regression_embeddings.embedding_to_words.nearest_volume_text_to_pred import get_best_and_worst_predictions


def main():
    embed_types = ['GloVe','BERT']
    filepath =Path(r'C:\Users\tibor\Documents\thesis\code_and_data\predictions\bayesian\per_participant')
    filename = 'bayes_vol_pred_sub-EN057_section8_mse_masked_Angular Gyrus_and_more.pkl'
    filename_embed = 'bayesian_embeddings_per_volume_sub-EN057.pkl'
    df = pd.read_pickle(filepath/filename_embed)

    for embed_type in embed_types:
        gt_emb = np.vstack(df[f'gt_{embed_type}'].values)
        gt_words = df['ground_truth'].tolist()
        pred_emb = np.vstack(df[f'pred_{embed_type}'].values)
        cosine_similarities = cosine_similarity(gt_emb, pred_emb)
        df[f'cosine_pred_{embed_type}'] = cosine_similarities.diagonal()


        # select nearest text
        # cosine_similarities = cosine_similarity(gt_embed, pred_emb)
        most_similar_index = np.argmax(cosine_similarities, axis=0)
        pred_words = [gt_words[i] for i in most_similar_index]
        df[f'pred_text_{embed_type}'] = pred_words
        df = df.rename(columns = {'ground_truth':'gt_text'})

        # print(f'for {participant}, and {embed_type}')
        get_best_and_worst_predictions(df=df, embed_type=embed_type)


    print('done')

if __name__ == '__main__':
    main()