import pandas as pd


def make_bold(val,tresh):
    if val>tresh:
        return f'{{\\bf{val}}}'
    else:
        return val
def analyze_baseline():
    dfbert = pd.read_pickle(r'F:\dataset\predictions\BERT\pred_embed_BERT_sub-EN057_section_8.pkl')
    dfglove = pd.read_pickle(r'F:\dataset\predictions\GloVe\pred_embed_GloVe_sub-EN057_section_8.pkl')

    print(f'mean glove: {dfglove.cosine_baseline_GloVe.mean()} std: {dfglove.cosine_baseline_GloVe.std()}')
    print(f'mean bert: {dfbert.cosine_baseline_BERT.mean()} std: {dfbert.cosine_baseline_BERT.std()}')

def analyze_two_participants():
    df = pd.read_csv(r'F:\dataset\predictions\overall\overview_predictions.csv')
    df = df[df['participant'].isin(['sub-EN057','sub-EN058'])]
    dfbert = df[df.embed_type=="BERT"]
    dfglove = df[df.embed_type=='GloVe']



def analyze_combined_regions():
    df = pd.read_csv(r'F:\dataset\predictions\overall\overview_predictions.csv')

    # round decimals
    df = df.round(decimals=6)
    baseline_GloVe = round(0.002532702616162057,6)
    baseline_BERT = round(-0.0035220872142616455,6)
    baseline_GloVe_std =round( 0.059781804096505346,6)
    baseline_BERT_std = round(0.03702856899118565,6)

    bert_regions = 'Angular Gyrus_Cerebelum_Crus1_L_Cerebelum_Crus1_R_Cerebelum_Crus2_L_Cerebelum_Crus2_R_Cerebelum_3_L_Cerebelum_3_R_Cerebelum_4_5_L_Cerebelum_4_5_R_Cerebelum_6_L_Cerebelum_6_R_Cerebelum_7b_L_Cerebelum_7b_R_Cerebelum_8_L_Cerebelum_8_R_Cerebelum_9_L_Cerebelum_9_R_Cerebelum_10_L_Cerebelum_10_R_Superior Temporal Gyrus, anterior division_Superior Temporal Gyrus, posterior division_Temporal_Sup_L_Temporal_Pole_Sup_L_Temporal_Mid_L_Temporal_Pole_Mid_L_Temporal_Inf_L_Temporal_Sup_R_Temporal_Pole_Sup_R_Temporal_Mid_R_Temporal_Pole_Mid_R_Temporal_Inf_R'
    glove_regions = 'Angular Gyrus_Cerebelum_Crus1_L_Cerebelum_Crus1_R_Cerebelum_Crus2_L_Cerebelum_Crus2_R_Cerebelum_3_L_Cerebelum_3_R_Cerebelum_4_5_L_Cerebelum_4_5_R_Cerebelum_6_L_Cerebelum_6_R_Cerebelum_7b_L_Cerebelum_7b_R_Cerebelum_8_L_Cerebelum_8_R_Cerebelum_9_L_Cerebelum_9_R_Cerebelum_10_L_Cerebelum_10_R_Superior Temporal Gyrus, anterior division_Superior Temporal Gyrus, posterior division_Temporal_Sup_R_Temporal_Pole_Sup_R_Temporal_Mid_R_Temporal_Pole_Mid_R_Temporal_Inf_R'

    dfbert = df[df.embed_type=="BERT"]
    dfglove = df[df.embed_type=='GloVe']

    dfbert = dfbert[dfbert.cortex_regions == bert_regions]
    dfbert.drop_duplicates(inplace=True)
    dfglove = dfglove[dfglove.cortex_regions == glove_regions]
    dfglove.drop_duplicates(inplace=True)


    # add average row
    # Calculate the means for numerical columns
    means_bert = dfbert[['cosine_similarity', 'cosine_similarity_std']].mean()
    # Create a new row with the participant value as "average" and means as scores
    average_row_bert = pd.DataFrame({'participant': ['Average'], **means_bert}).round(6)
    dfbert = pd.concat([dfbert, average_row_bert])
    means_glove = dfbert[['cosine_similarity', 'cosine_similarity_std']].mean()
    # Create a new row with the participant value as "average" and means as scores
    average_row_glove = pd.DataFrame({'participant': ['Average'], **means_glove}).round(6)
    dfglove = pd.concat([dfglove, average_row_glove])

    #add baseline
    dfbert = pd.concat([
        dfbert,
        pd.DataFrame([{'participant':'Baseline','cosine_similarity': baseline_BERT,'cosine_similarity_std':baseline_BERT_std}])])
    dfglove = pd.concat([
        dfglove,
        pd.DataFrame([{'participant':'Baseline','cosine_similarity': baseline_GloVe,'cosine_similarity_std':baseline_GloVe_std}])])

    # select above baseline
    # dfbert = dfbert[dfbert.cosine_similarity >= baseline_BERT]
    # dfglove = dfglove[dfglove.cosine_similarity >= baseline_GloVe]
    #make bold if above baseline
    dfglove['cosine_similarity'] = dfglove['cosine_similarity'].apply(
        make_bold,
        args=(baseline_GloVe,))
    dfbert['cosine_similarity'] = dfbert['cosine_similarity'].apply(
        make_bold,
        args=(baseline_BERT,))


    # format tables and make latex
    dfglove = dfglove[['participant','cosine_similarity', 'cosine_similarity_std']]
    dfbert = dfbert[['participant','cosine_similarity', 'cosine_similarity_std']]
    # sort values by participants
    dfglove = dfglove.sort_values(by='participant',ascending=False)
    dfbert = dfbert.sort_values(by='participant',ascending=False)

    # rename columns
    dfbert = dfbert.rename(columns = {
        'cosine_similarity':'cos sim bert',
        'cosine_similarity_std':'std cos sim bert'
    })
    dfglove = dfglove.rename(columns = {
        'cosine_similarity':'cos sim glove',
        'cosine_similarity_std':'std cos sim glove'
    })
    # print(dfbert.to_latex(index=False))
    dfnew = pd.merge(dfglove,dfbert, on='participant')

    # dfnew = pd.concat([dfglove,dfbert],axis =1)
    latex_string = dfnew.to_latex(index=False)
    latex_string = latex_string.replace('llrlr', 'lrrrr')
    # latex_string.replace('std cos sim bert', 'std cos sim')
    # latex_string.replace('std cos sim glove', 'std cos sim')
    latex_string = latex_string.replace('bert', '')
    latex_string = latex_string.replace('glove', '')
    #insert nested column header
    insert = ''' & \multicolumn{2}{c}{GloVe} & \multicolumn{2}{c}{BERT} \\\\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}'''
    latex_string = latex_string.replace('\\toprule', '\\toprule' + insert, 1)

    print(latex_string)
    # print(dfnew.to_latex(index=False))
    print('done')


def get_unique_regions(df):
    unique_regions_by_participant = df.groupby('participant')['cortex_regions'].unique()

    # Find the intersection of unique regions for all participants
    common_regions = set(unique_regions_by_participant.values[0]).intersection(*unique_regions_by_participant.values)

    # Convert the common regions to a list
    common_regions_list = list(common_regions)
    return common_regions_list
def main():
    analyze_combined_regions()
    df = pd.read_csv(r'F:\dataset\predictions\overall\overview_predictions.csv')
    baseline_GloVe = 0.002532702616162057
    baseline_BERT = -0.0035220872142616455

    dfbert = df[df.embed_type=="BERT"]
    dfglove = df[df.embed_type=='GloVe']

    dfbert = dfbert[dfbert.cosine_similarity >= baseline_BERT]
    dfglove = dfglove[dfglove.cosine_similarity >= baseline_GloVe]

    # sorted_df = df.sort_values(by=['cosine_similarity','participant', 'embed_type'],ascending=True)
    sorted_df = df.sort_values(by=['participant', 'embed_type', 'cosine_similarity'],ascending=True)
    sorted_df = df.sort_values(by=['participant', 'embed_type', 'cosine_similarity'], ascending=[True, True, False])

    # select unique above baseline regions
    bert_regions = dfbert.cortex_regions.unique()
    bert_regions.sort()
    l = [b for b in bert_regions]
    flattened_list = [item for sublist in bert_regions for item in sublist]

    bert_regions = [b for bl in bert_regions for b in bl]
    flattened_list = [item for sublist in array_of_lists for item in sublist]

    # select intersection of regions above baseline

    dfbert = dfbert[df['participant'].isin(['sub-EN057','sub-EN058'])]
    regions_above_baseline_bert = dfbert.cortex_regions.unique()
    regions_above_baseline_glove = dfglove.cortex_regions.unique()
    significant_regions_bert = get_unique_regions(dfbert)
    significant_regions_glove = get_unique_regions(dfglove)

    print('end')


if __name__ == '__main__':
    main()