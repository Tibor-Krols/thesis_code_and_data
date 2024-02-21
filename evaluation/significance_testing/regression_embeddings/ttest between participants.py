import pandas as pd
from scipy import stats
from utils.paths import *
from scipy.stats import shapiro
from scipy.stats import levene


def test_ttest_assumptions(df,embed_type):
    # normality:
    # Shapiro-Wilk test for normality for group1
    statistic, p_value = shapiro(df[f'cosine_pred_{embed_type}'])
    print("Shapiro-Wilk test for normality pred:")
    print("Test Statistic:", statistic)
    print("p-value:", p_value)

    # Shapiro-Wilk test for normality for group2
    statistic, p_value = shapiro(df[f'cosine_baseline_{embed_type}'])
    print("Shapiro-Wilk test for normality baseline:")
    print("Test Statistic:", statistic)
    print("p-value:", p_value)


    # Levene's test for homogeneity of variances
    statistic, p_value = levene(df[f'cosine_pred_{embed_type}'], df[f'cosine_baseline_{embed_type}'])
    print("Levene's test for homogeneity of variances:")
    print("Test Statistic:", statistic)
    print("p-value:", p_value)




def assign_seen_unseen(
        df,
        train_participants = ['sub-EN077', 'sub-EN069', 'sub-EN099', 'sub-EN086', 'sub-EN104'],
        train_sections = [1, 3, 5, 8]

):
    # assign seen and unseen
    df['all'] = True
    df['unseen_participant'] = [(p not in train_participants) and (s in train_sections) for p,s in zip(df.participant,df.section)]
    df['unseen_section'] = [(p in train_participants) and (s not in train_sections) for p,s in zip(df.participant,df.section)]
    df['fully_unseen'] = [(p not in train_participants) and (s not in train_sections) for p,s in zip(df.participant,df.section)]
    df['fully_seen'] = [(p in train_participants) and (s in train_sections) for p,s in zip(df.participant,df.section)]

    return df


def ttest_between_participants():
    embed_types = ['GloVe','BERT']


    dfstat = pd.DataFrame()
    partitions = ['all','unseen_participant','unseen_section','fully_unseen','fully_seen']
    for embed_type in embed_types:
        p_embed = []
        t_embed = []
        filepath = Path(fr'F:\dataset\predictions\across_participants\{embed_type}\train_participants_sub-EN077_sub-EN069_sub-EN099_sub-EN086_sub-EN104_train_sections_1_3_5_8')
        filename = f'pred_embed_{embed_type}_test_sub-EN086_sub-EN104_sub-EN101_sub-EN106_test_sections_2_8_Angular Gyrus_and_more.pkl'
        df = pd.read_pickle(filepath/filename)
        df = assign_seen_unseen(df)
        for partition in partitions:
            df = pd.read_pickle(filepath/filename)
            df = assign_seen_unseen(df)
            df = df[df[partition]]
            # mean_pred= df[f'cosine_pred_{embed_type}'].mean()
            print(f"{embed_type} {partition} (M={round(df[f'cosine_pred_{embed_type}'].mean(),2)}, SD={round(df[f'cosine_pred_{embed_type}'].std(),2)})")
            print(f"Baseline (M={round(df[f'cosine_baseline_{embed_type}'].mean(),2)}, SD={round(df[f'cosine_baseline_{embed_type}'].std(),2)})")
            print('\n')
            # Perform t-test
            t_statistic, p_value = stats.ttest_ind(
                df[f'cosine_pred_{embed_type}'],
                df[f'cosine_baseline_{embed_type}'],
                equal_var=False
            )
            p_embed.append(p_value)
            t_embed.append(t_statistic)
        dfstat['partition']=partitions
        dfstat[f'P_value_{embed_type}'] = p_embed
        dfstat[f't_statistic_{embed_type}'] = t_embed
        test_ttest_assumptions(df, embed_type)


    dfstat = dfstat.sort_values(by=['P_value_GloVe', 'P_value_BERT'])


    alpha = 0.05
    bonferroni_corr_alpha = alpha /dfstat.shape[0]
    dfsigglove = dfstat[dfstat.P_value_GloVe < bonferroni_corr_alpha]
    dfsigbert = dfstat[dfstat.P_value_BERT < bonferroni_corr_alpha]

    return dfstat



def main():

    embed_type = 'BERT'
    participant = 'sub-EN057'
    filename = f'pred_embed_{embed_type}_{participant}_section_8_Angular Gyrus_and_more.pkl'
                 # 'pred_embed_GloVe_sub-EN057_section_8_.pkl'

    # load data
    df = pd.read_pickle(data_path/'predictions'/embed_type/filename)

    ttest_between_participants()
    # do ttest
    # Perform t-test
    t_statistic, p_value = stats.ttest_ind(
        df[f'cosine_pred_{embed_type}'],
        df[f'cosine_baseline_{embed_type}']
    )

    # Print results
    print("t-statistic:", t_statistic)
    print("p-value:", p_value)

    pass


if __name__ == '__main__':
    main()