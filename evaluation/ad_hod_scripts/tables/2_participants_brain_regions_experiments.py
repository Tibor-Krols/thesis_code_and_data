
import pandas as pd

from evaluation.ad_hod_scripts.analyze_overall_scores_word_embeddings import make_bold

def analyze_baseline():
    dfglove = pd.read_csv(r'C:\Users\tibor\Documents\thesis\code_and_data\predictions\baseline\embeddings\baseline_predictions_per_volume_full_text.csv')
    dfglove = pd.read_pickle(r'F:\dataset\predictions\BERT\pred_embed_BERT_sub-EN057_section_8.pkl')


def add_average_2_participants(df):
    # Calculate mean cosine similarity and mean cosine similarity std over 2 participants
    mean_cosine_similarity = df.groupby('cortex_regions')['cosine_similarity'].mean()
    mean_cosine_similarity_std = df.groupby('cortex_regions')['cosine_similarity_std'].mean()

    # Create new DataFrame with average values
    new_data = pd.DataFrame({
        'participant': ['Average'] * 7,
        'cortex_regions': mean_cosine_similarity.index,
        'cosine_similarity': mean_cosine_similarity.values,
        'cosine_similarity_std': mean_cosine_similarity_std.values
    })
    df = pd.concat([df,new_data])
    df = df.round(decimals=6)
    return df
def analyze_two_participants():
    df = pd.read_csv(r'F:\dataset\predictions\overall\overview_predictions.csv')
    df = df[df['participant'].isin(['sub-EN057','sub-EN058'])]

    df = df.round(decimals=6)

    # brain regions
    angular_gyrus = 'Angular Gyrus'
    IFG = 'Inferior Frontal Gyrus, pars triangularis_Inferior Frontal Gyrus, pars opercularis'
    precuneus = 'Precuneous Cortex'
    cerebellum = 'Cerebelum_Crus1_L_Cerebelum_Crus1_R_Cerebelum_Crus2_L_Cerebelum_Crus2_R_Cerebelum_3_L_Cerebelum_3_R_Cerebelum_4_5_L_Cerebelum_4_5_R_Cerebelum_6_L_Cerebelum_6_R_Cerebelum_7b_L_Cerebelum_7b_R_Cerebelum_8_L_Cerebelum_8_R_Cerebelum_9_L_Cerebelum_9_R_Cerebelum_10_L_Cerebelum_10_R'
    STG = 'Superior Temporal Gyrus, anterior division_Superior Temporal Gyrus, posterior division'
    L_temp = 'Temporal_Sup_L_Temporal_Pole_Sup_L_Temporal_Mid_L_Temporal_Pole_Mid_L_Temporal_Inf_L'
    R_temp = 'Temporal_Sup_R_Temporal_Pole_Sup_R_Temporal_Mid_R_Temporal_Pole_Mid_R_Temporal_Inf_R'

    # only select individual brain regions
    df = df[df['cortex_regions'].isin([angular_gyrus, IFG, precuneus, cerebellum, STG,L_temp,R_temp])]

    #fix brain regions names
    df = df.replace(IFG,'Inferior Frontal Gyrus')
    df = df.replace(precuneus, 'Precuneus')
    df = df.replace(cerebellum,'Cerebellum')
    df = df.replace(STG,'Superior Temporal Gyrus')
    df = df.replace(L_temp, 'Left Temporal Lobe')
    df = df.replace(R_temp, 'Right Temporal Lobe')

    # split_bert and glove
    dfbert = df[df.embed_type=="BERT"]
    dfglove = df[df.embed_type=='GloVe']

    #sort by participants and cortex
    dfbert = dfbert.sort_values(by=['participant','cortex_regions'],ignore_index=True)
    dfglove = dfglove.sort_values(by=['participant','cortex_regions'],ignore_index=True)


    # add average over 2 participants
    # dfavg = dfglove.groupby(by=['cortex_regions'])['cosine_similarity'].mean()
    dfglove = add_average_2_participants(dfglove)
    dfbert =  add_average_2_participants(dfbert)

    # make bold above baseline
    baseline_GloVe = round(0.002532702616162057,6)
    baseline_BERT = round(-0.0035220872142616455,6)
    baseline_GloVe_std =round( 0.059781804096505346,6)
    baseline_BERT_std = round(0.03702856899118565,6)
    dfglove['cosine_similarity'] = dfglove['cosine_similarity'].apply(
        make_bold,
        args=(baseline_GloVe,))
    dfbert['cosine_similarity'] = dfbert['cosine_similarity'].apply(
        make_bold,
        args=(baseline_BERT,))

    # format tables and make latex
    dfglove = dfglove[['participant','cortex_regions','cosine_similarity', 'cosine_similarity_std']]
    dfbert = dfbert[['participant','cortex_regions','cosine_similarity', 'cosine_similarity_std']]


    # add baseline row to tables
    dfglove.loc[len(dfglove.index)] = ['Baseline','',baseline_GloVe,baseline_GloVe_std]
    dfbert.loc[len(dfbert.index)] = ['Baseline','', baseline_BERT,baseline_BERT_std]

    # rename columns
    dfbert = dfbert.rename(columns = {
        'cosine_similarity':'cos sim bert',
        'cosine_similarity_std':'std cos sim bert'
    })
    dfglove = dfglove.rename(columns = {
        'cosine_similarity':'cos sim glove',
        'cosine_similarity_std':'std cos sim glove'
    })
    dfnew = pd.merge(dfglove,dfbert, on=['participant','cortex_regions'])
    dfnew = dfnew.drop_duplicates()

    # format latex
    latex_string = dfnew.to_latex(index=False)
    latex_string = latex_string.replace('llrlr', 'lrrrr')
    latex_string = latex_string.replace('bert', '')
    latex_string = latex_string.replace('glove', '')
    # insert nested column header
    insert = ''' & \multicolumn{2}{r}{GloVe} & \multicolumn{2}{r}{BERT} \\\\
    \cmidrule(lr){3-4} \cmidrule(lr){5-6}'''
    latex_string = latex_string.replace('\\toprule', '\\toprule' + insert, 1)
    latex_string = latex_string.replace('_','\\_')
    print(latex_string)
    print('done')


if __name__ == '__main__':
    analyze_two_participants()