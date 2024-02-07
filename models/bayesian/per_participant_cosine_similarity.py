from pathlib import Path

import numpy as np
import pandas as pd

from dataset_loader.dataset import LPPDataset
from evaluation.ad_hod_scripts.analyze_overall_scores_word_embeddings import make_bold
from models.regression_embeddings.embedding_to_words.nearest_volume_text_to_pred import load_embeddings_section

from sklearn.metrics.pairwise import cosine_similarity

def add_gt_embeddings():
    participants = ['sub-EN057', 'sub-EN058']
    section = 8
    embed_types = ['GloVe', 'BERT']
    for participant in participants:
        # load predictions
        filepath = Path(r'C:\Users\tibor\Documents\thesis\code_and_data\predictions\bayesian\per_participant')
        filename = f'bayes_vol_pred_{participant}_section{section}_mse_masked_Angular Gyrus_and_more.csv'
        dfpred = pd.read_csv(filepath / filename)
        dfpred = dfpred.dropna()

        # add baseline embeddings to predictions
        # for both embeddings
        for embed_type in embed_types:
            dataset = LPPDataset(embed_type=embed_type)

            gt_embed, gt_words = load_embeddings_section(dataset=dataset,
                                                         embed_type=embed_type,
                                                         participant=participant,
                                                         section=section)
            dfpred[f'gt_{embed_type}'] = gt_embed.tolist()

        filename = filename[:-4] + '.pkl'  # change .csv to .pkl
        dfpred.to_pickle(filepath / filename)


def make_table(df):
    embed_types = ['GloVe','BERT']
    df = df.round(decimals=6)

    # make bold above baseline
    baseline_GloVe = round(0.002532702616162057, 6)
    baseline_BERT = round(-0.0035220872142616455, 6)
    baseline_GloVe_std = round(0.059781804096505346, 6)
    baseline_BERT_std = round(0.03702856899118565, 6)
    # add baseline
    new_row_values = ['Baseline', baseline_GloVe,baseline_GloVe_std,baseline_BERT,baseline_BERT_std]
    df.loc[len(df.index)] = new_row_values

    # make above baseline bold
    df[f'cos_sim_GloVe'] = df[f'cos_sim_GloVe'].apply(
        make_bold,
        args=(baseline_GloVe,))

    df[f'cos_sim_BERT'] = df[f'cos_sim_BERT'].apply(
        make_bold,
        args=(baseline_BERT,))

    # make latex table
    df.columns = df.columns.str.replace('_', ' ')

    latex_string = df.to_latex(index=False)
    latex_string = latex_string.replace('llrlr', 'lrrrr')
    # latex_string.replace('std cos sim bert', 'std cos sim')
    # latex_string.replace('std cos sim glove', 'std cos sim')
    latex_string = latex_string.replace('BERT', '')
    latex_string = latex_string.replace('GloVe', '')
    # insert nested column header
    insert = ''' & \multicolumn{2}{c}{GloVe} & \multicolumn{2}{c}{BERT} \\\\
    \cmidrule(lr){2-3} \cmidrule(lr){4-5}'''
    latex_string = latex_string.replace('\\toprule', '\\toprule' + insert, 1)

    print(latex_string)
    return df

def main():
    participants = ['sub-EN057', 'sub-EN058']
    section = 8
    embed_types = ['GloVe', 'BERT']

    filepath = Path(r'C:\Users\tibor\Documents\thesis\code_and_data\predictions\bayesian\per_participant')

    dfres = pd.DataFrame()
    dfres['participant'] = participants
    for embed_type in embed_types:
        cos_sim = []
        cos_sim_std = []
        for participant in participants:
            filename = f'bayesian_embeddings_per_volume_{participant}.pkl'
            df = pd.read_pickle(filepath / filename)

            gt_embed = np.vstack(df[f'gt_{embed_type}'].values)
            pred_emb = np.vstack(df[f'pred_{embed_type}'].values)

            # get cosine
            cosine_similarities = cosine_similarity(gt_embed, pred_emb)
            cosine_similarities = np.diagonal(cosine_similarities, axis1=0, axis2=1)
            cos_sim.append(cosine_similarities.mean())
            cos_sim_std.append(cosine_similarities.std())
        dfres[f'cos_sim_{embed_type}'] = cos_sim
        dfres[f'cos_sim_std_{embed_type}'] = cos_sim_std
    make_table(dfres)
    print('done')

if __name__ == '__main__':
    main()