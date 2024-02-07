import pandas as pd

from evaluation.ad_hod_scripts.analyze_overall_scores_word_embeddings import make_bold
from utils.paths import *

def main():
    train_participants = ['sub-EN077', 'sub-EN069', 'sub-EN099', 'sub-EN086', 'sub-EN104']
    train_sections = [1, 3, 5, 8]
    test_participants = ['sub-EN086', 'sub-EN104', 'sub-EN101', 'sub-EN106']
    test_sections = [2, 8]

    # calculate means
    # embed_type = "BERT"
    dfbert = create_results_df(
        embed_type="BERT",
        train_participants=train_participants,
        train_sections=train_sections,
        test_participants=test_participants,
        test_sections=test_sections
    )
    dfglove = create_results_df(
        embed_type="GloVe",
        train_participants=train_participants,
        train_sections=train_sections,
        test_participants=test_participants,
        test_sections=test_sections
    )

    # rename columns
    dfbert = dfbert.rename(columns = {
        'cosine_similarity':'cos sim bert',
        'cosine_similarity_std':'std cos sim bert'
    })
    dfglove = dfglove.rename(columns = {
        'cosine_similarity':'cos sim glove',
        'cosine_similarity_std':'std cos sim glove'
    })
    dfnew = pd.merge(dfglove,dfbert, on=['test_data'])
    dfnew = dfnew.drop_duplicates()


    # format latex
    latex_string = dfnew.to_latex(index=False)
    latex_string = latex_string.replace('llrlr', 'lrrrr')
    latex_string = latex_string.replace('bert', '')
    latex_string = latex_string.replace('glove', '')
    # insert nested column header
    insert = ''' & \multicolumn{2}{c}{GloVe} & \multicolumn{2}{c}{BERT} \\\\
    \cmidrule(lr){2-3} \cmidrule(lr){4-5}'''
    latex_string = latex_string.replace('\\toprule', '\\toprule' + insert, 1)
    latex_string = latex_string.replace('_','\\_')
    print(latex_string)

    pass




def create_results_df(
        embed_type:str,
        train_participants,
        train_sections,
        test_participants,
        test_sections
) -> pd.DataFrame:
    train_participants_string = '_'.join(train_participants)
    train_sections_string = '_'.join([str(s) for s in train_sections])
    test_participants_string = '_'.join(test_participants)
    test_sections_string = '_'.join([str(s) for s in test_sections])
    filepath = data_path/'predictions'/'across_participants'/embed_type/ f'train_participants_{train_participants_string}_train_sections_{train_sections_string}'
    filename = f'pred_embed_{embed_type}_test_{test_participants_string}_test_sections_{test_sections_string}_Angular Gyrus_and_more.pkl'
    df = pd.read_pickle(filepath/filename)
    # df = df.round(decimals=6)

    # baseline TODO: chose baseline to use
    baseline = df[f'cosine_baseline_{embed_type}'].mean()
    baseline_std = df[f'cosine_baseline_{embed_type}'].std()
    # if embed_type=='BERT':
    #     baseline = round(-0.0035220872142616455, 6)
    #     baseline_std = round(0.03702856899118565, 6)
    # if embed_type =='GloVe':
    #     baseline = round(0.002532702616162057, 6)
    #     baseline_std = round(0.059781804096505346, 6)


    # assign seen and unseen
    df['unseen_participant'] = [(p not in train_participants) and (s in train_sections) for p,s in zip(df.participant,df.section)]
    df['unseen_section'] = [(p in train_participants) and (s not in train_sections) for p,s in zip(df.participant,df.section)]
    df['fully_unseen'] = [(p not in train_participants) and (s not in train_sections) for p,s in zip(df.participant,df.section)]
    df['fully_seen'] = [(p in train_participants) and (s in train_sections) for p,s in zip(df.participant,df.section)]


    columns = ['unseen_participant','unseen_section','fully_unseen','fully_seen']
    results_dict = {'test_data':[],'cosine_similarity':[],'cosine_similarity_std':[]}
    results_dict['test_data'].append('all')
    results_dict['cosine_similarity'].append(
        df[f'cosine_pred_{embed_type}'].mean()
    )
    results_dict['cosine_similarity_std'].append(
        df[f'cosine_pred_{embed_type}'].std()
    )
    for col in columns:
        results_dict['test_data'].append(col)
        results_dict['cosine_similarity'].append(
            df[df[col]][f'cosine_pred_{embed_type}'].mean()
        )
        results_dict['cosine_similarity_std'].append(
            df[df[col]][f'cosine_pred_{embed_type}'].std()
        )
    #add baseline
    results_dict['test_data'].append('Baseline')
    results_dict['cosine_similarity'].append(
        baseline
    )
    results_dict['cosine_similarity_std'].append(
        baseline_std
    )

    dfres = pd.DataFrame(results_dict)
    dfres = dfres.round(decimals=6)
    # # make above baseline_bold
    dfres['cosine_similarity'] = dfres['cosine_similarity'].apply(
        make_bold,
        args=(baseline,))

    return dfres








if __name__ == '__main__':
    main()
    # check different kinds of unseen
    # mean_full = df[df['fully_unseen']][f'cosine_pred_{embed_type}'].mean()
    # print(f'fully unseen mean cosine {mean_full}')
    #
    # mean_unseen_participant =     df[
    #     df['unseen_participant'] & ~df['unseen_section']
    # ][f'cosine_pred_{embed_type}'].mean()
    # print(f'unseen participant, seen section: {mean_unseen_participant}')
    #
    # print(df[df['unseen_participant']][f'cosine_pred_{embed_type}'].mean(),
    # df[df['unseen_section']][f'cosine_pred_{embed_type}'].mean(),
    # df[df['fully_unseen']][f'cosine_pred_{embed_type}'].mean())