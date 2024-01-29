import pandas as pd
from scipy import stats
from utils.paths import *



def ttest_individual_brain_regions():
    pass
def ttest_multiple_brain_regions():
    participants = ['sub-EN115', 'sub-EN114', 'sub-EN113', 'sub-EN110', 'sub-EN100', 'sub-EN097', 'sub-EN096', 'sub-EN094', 'sub-EN092', 'sub-EN091', 'sub-EN089', 'sub-EN088', 'sub-EN086', 'sub-EN084', 'sub-EN082', 'sub-EN081', 'sub-EN076', 'sub-EN075', 'sub-EN070', 'sub-EN068', 'sub-EN067', 'sub-EN064', 'sub-EN063', 'sub-EN062', 'sub-EN059', 'sub-EN058', 'sub-EN057']
    embed_types = ['GloVe','BERT']


    dfstat = pd.DataFrame()
    dfstat['participant']= participants
    for embed_type in embed_types:
        p_embed = []
        t_embed = []
        for participant in participants:
            filename = f'pred_embed_{embed_type}_{participant}_section_8_Angular Gyrus_and_more.pkl'
            # load data
            df = pd.read_pickle(data_path/'predictions'/embed_type/filename)

            # do ttest
            # Perform t-test
            t_statistic, p_value = stats.ttest_rel(
                df[f'cosine_pred_{embed_type}'],
                df[f'cosine_baseline_{embed_type}']
            )
            p_embed.append(p_value)
            t_embed.append(t_statistic)
        dfstat[f'P_value_{embed_type}'] = p_embed
        dfstat[f't_statistic_{embed_type}'] = t_embed


    dfstat = dfstat.sort_values(by = ['P_value_GloVe','P_value_BERT'])



def main():

    embed_type = 'BERT'
    participant = 'sub-EN057'
    filename = f'pred_embed_{embed_type}_{participant}_section_8_Angular Gyrus_and_more.pkl'
                 # 'pred_embed_GloVe_sub-EN057_section_8_.pkl'

    # load data
    df = pd.read_pickle(data_path/'predictions'/embed_type/filename)

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