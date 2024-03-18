from pathlib import Path

import numpy as np
import pandas as pd

from models.bayesian.per_participant_cosine_similarity import add_cosine_to_bayesian_df
from sklearn.metrics.pairwise import cosine_similarity

from models.regression_embeddings.embedding_to_words.nearest_volume_text_to_pred import get_best_and_worst_predictions

def get_best_and_worst_predictions_bayesian(df,embed_type):
    df = df.sort_values(by=f'cosine_pred_{embed_type}',ascending=False)
    if 'gt_text' not in df.columns:
        df['gt_text'] = df[f'volume_words'].apply(lambda x: ' '.join(x))

    # df[f'pred_text_{embed_type}'] = df[f'pred_words'].apply(lambda x: ' '.join(x))

    df = df[['gt_text','pred_text',f'pred_text_{embed_type}',f'cosine_pred_{embed_type}']]


    df = df.rename(columns={f'cosine_pred_{embed_type}':'cos sim'})
    dfnew = pd.concat([df.head(10),df.tail(10)])
    dfnew = dfnew.rename(columns = {
        'gt_text': 'Ground Truth',
        'pred_text': 'Bayesian Prediction',
        f'pred_text_{embed_type}': 'Nearest Neighbour'
    })

    latex_string = dfnew.to_latex(index=False)
    latex_string = latex_string.replace('_','\\_')


    print(latex_string)
def main():
    embed_types = ['GloVe','BERT']
    filepath =Path(r'C:\Users\tibor\Documents\thesis\code_and_data\predictions\bayesian\per_participant')
    filename = 'bayes_vol_pred_sub-EN057_section8_mse_masked_Angular Gyrus_and_more.pkl'
    filename_embed = 'bayesian_embeddings_per_volume_sub-EN057.pkl'
    df = pd.read_pickle(filepath/filename_embed)
    gt_words = df['ground_truth'].tolist()
    df = df.rename(columns = {'ground_truth':'gt_text'})

    for embed_type in embed_types:
        gt_emb = np.vstack(df[f'gt_{embed_type}'].values)
        pred_emb = np.vstack(df[f'pred_{embed_type}'].values)
        cosine_similarities = cosine_similarity(gt_emb, pred_emb)
        df[f'cosine_pred_{embed_type}'] = cosine_similarities.diagonal()


        # select nearest text
        # cosine_similarities = cosine_similarity(gt_embed, pred_emb)
        most_similar_index = np.argmax(cosine_similarities, axis=0)
        pred_words = [gt_words[i] for i in most_similar_index]
        df[f'pred_text_{embed_type}'] = pred_words

        # print(f'for {participant}, and {embed_type}')
        print(embed_type)
        get_best_and_worst_predictions_bayesian(df=df, embed_type=embed_type)


    print('done')

if __name__ == '__main__':
    main()