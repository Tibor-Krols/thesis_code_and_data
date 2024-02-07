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






def ttest_individual_brain_regions():
    pass
def ttest_multiple_brain_regions():
    participants = ['sub-EN115', 'sub-EN114', 'sub-EN113', 'sub-EN110', 'sub-EN100', 'sub-EN097', 'sub-EN096', 'sub-EN094', 'sub-EN092', 'sub-EN091', 'sub-EN089', 'sub-EN088', 'sub-EN086', 'sub-EN084', 'sub-EN082', 'sub-EN081', 'sub-EN076', 'sub-EN075', 'sub-EN070', 'sub-EN068', 'sub-EN067', 'sub-EN064', 'sub-EN063', 'sub-EN062', 'sub-EN059', 'sub-EN058', 'sub-EN057']
    embed_types = ['GloVe','BERT']


    dfstat = pd.DataFrame()
    # dfstat['participant']= participants
    for embed_type in embed_types:
        p_embed = []
        t_embed = []
        df_embed = pd.DataFrame()
        for participant in participants:
            filename = f'pred_embed_{embed_type}_{participant}_section_8_Angular Gyrus_and_more.pkl'
            # load data
            df = pd.read_pickle(data_path/'predictions'/embed_type/filename)
            df_embed = pd.concat([df_embed,df])
        # do ttest
        # Perform t-test
        t_statistic, p_value = stats.ttest_ind(
            df_embed[f'cosine_pred_{embed_type}'],
            df_embed[f'cosine_baseline_{embed_type}'],
            equal_var=False
        )
        p_embed.append(p_value)
        t_embed.append(t_statistic)
        dfstat[f'P_value_{embed_type}'] = p_embed
        dfstat[f't_statistic_{embed_type}'] = t_embed
        test_ttest_assumptions(df_embed, embed_type)


    dfstat = dfstat.sort_values(by = ['P_value_GloVe','P_value_BERT'])



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

    ttest_multiple_brain_regions()
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