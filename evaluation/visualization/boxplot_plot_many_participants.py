import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from evaluation.ad_hod_scripts.analyze_overall_scores_word_embeddings import analyze_combined_regions
from evaluation.significance_testing.regression_embeddings.ttest_avg_many_participants import load_df_embed
from utils.paths import eval_path


def make_boxplot_27_participants(df):
    # preprocess data

    baseline_glove = round(0.002532702616162057,6)
    baseline_bert = round(-0.0035220872142616455,6)
    baseline_glove_std = round( 0.059781804096505346,6)
    baseline_bert_std = round(0.03702856899118565,6)
    # baseline_bert = df[df['participant']== 'Baseline']['cos sim bert'].iloc[0]
    # baseline_glove = df[df['participant']== 'Baseline']['cos sim glove'].iloc[0]
    # df = df[df['participant'] != 'Average']
    # df = df[df['participant'] != 'Baseline']
    df = df[['cos sim glove', 'cos sim bert']]
    df = df.rename(columns = {'cos sim glove':'GloVe', 'cos sim bert':'BERT'})

    #make plot
    # Set up subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # Plot boxplot for BERT
    sns.boxplot( data=df['BERT'], ax=axes[0],color='orange')
    axes[0].axhline(y=baseline_bert, color='r', linestyle='--',linewidth =2, label='Baseline')
    axes[0].set_title('BERT',fontsize = 'xx-large')
    # axes[0].set_xlabel('Cosine Similarity')
    axes[0].set_ylabel('Cosine Similarity with Ground Truth',fontsize = 'xx-large')
    axes[0].legend(fontsize = 'x-large')
    axes[0].set_xticks([])  # Remove ticks from x-axis

    # Plot boxplot for GloVe
    sns.boxplot(data=df['GloVe'], ax=axes[1])
    axes[1].axhline(y=baseline_glove, color='r', linestyle='--',linewidth = 2, label='Baseline')
    axes[1].set_title('GloVe',fontsize = 'xx-large')
    # axes[1].set_xlabel('Cosine Similarity')
    axes[1].set_ylabel('Cosine Similarity with Ground Truth',fontsize = 'xx-large')
    axes[1].legend(fontsize = 'x-large')
    axes[1].set_xticks([])  # Remove ticks from x-axis

    # Set same y axis scale for both plots
    max_y = max([max(df['BERT']), max(df['GloVe']), baseline_bert, baseline_glove])
    min_y = min([min(df['BERT']), min(df['GloVe']), baseline_bert, baseline_glove])
    # min_y = -0.02
    # max_y = 0.08
    axes[0].set_ylim(min_y, max_y)
    axes[1].set_ylim(min_y, max_y)

    plt.tight_layout()
    # plt.show()
    return plt
def make_boxplot(df):
    # preprocess data
    baseline_bert = df[df['participant']== 'Baseline']['cos sim bert'].iloc[0]
    baseline_glove = df[df['participant']== 'Baseline']['cos sim glove'].iloc[0]
    df = df[df['participant'] != 'Average']
    df = df[df['participant'] != 'Baseline']
    df = df[['cos sim glove', 'cos sim bert']]
    df = df.rename(columns = {'cos sim glove':'GloVe', 'cos sim bert':'BERT'})

    #make plot
    # Set up subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # Plot boxplot for BERT
    sns.boxplot( data=df['BERT'], ax=axes[0],color='orange',showmeans=True, meanprops={"marker":"x", "markerfacecolor":"red", "markeredgecolor":"red"})
    axes[0].axhline(y=baseline_bert, color='r', linestyle='--',linewidth =2, label='Baseline')
    axes[0].set_title('BERT',fontsize = 'xx-large')
    # axes[0].set_xlabel('Cosine Similarity')
    axes[0].set_ylabel('Cosine Similarity with Ground Truth',fontsize = 'xx-large')
    axes[0].set_xticks([])  # Remove ticks from x-axis

    # Plot boxplot for GloVe
    sns.boxplot(data=df['GloVe'], ax=axes[1],showmeans=True, meanprops={"marker":"x", "markerfacecolor":"red", "markeredgecolor":"red"})
    axes[1].axhline(y=baseline_glove, color='r', linestyle='--',linewidth = 2, label='Baseline')
    axes[1].set_title('GloVe',fontsize = 'xx-large')
    # axes[1].set_xlabel('Cosine Similarity')
    axes[1].set_ylabel('Cosine Similarity with Ground Truth',fontsize = 'xx-large')
    # Custom legend
    handles = [Line2D([0], [0], marker='x', color='red', markerfacecolor='red', markersize=10, linestyle='None'),
               Line2D([0], [0], color='red', linestyle='--', linewidth=2)]
    labels = ['Mean', 'Baseline']
    axes[0].legend(handles, labels, fontsize='x-large')
    axes[1].legend(handles, labels, fontsize='x-large')
    axes[1].set_xticks([])  # Remove ticks from x-axis

    # Set same y axis scale for both plots
    max_y = max([max(df['BERT']), max(df['GloVe']), baseline_bert, baseline_glove])
    min_y = min([min(df['BERT']), min(df['GloVe']), baseline_bert, baseline_glove])
    min_y = -0.02
    max_y = 0.08
    axes[0].set_ylim(min_y, max_y)
    axes[1].set_ylim(min_y, max_y)

    plt.tight_layout()
    # plt.show()
    return plt
def save_boxplot(plt,filename = 'boxplot_27_participants.png' ):
    filepath = eval_path /'visualization'/'plots'
    plt.savefig(filepath/filename)

def get_27_participants_df():
    participants = ['sub-EN115', 'sub-EN114', 'sub-EN113', 'sub-EN110', 'sub-EN100', 'sub-EN097', 'sub-EN096', 'sub-EN094', 'sub-EN092', 'sub-EN091', 'sub-EN089', 'sub-EN088', 'sub-EN086', 'sub-EN084', 'sub-EN082', 'sub-EN081', 'sub-EN076', 'sub-EN075', 'sub-EN070', 'sub-EN068', 'sub-EN067', 'sub-EN064', 'sub-EN063', 'sub-EN062', 'sub-EN059', 'sub-EN058', 'sub-EN057']
    embed_types = ['GloVe','BERT']

    df = pd.DataFrame()
    for embed_type in embed_types:
        df_embed = load_df_embed(embed_type=embed_type, participants=participants)
        df[f'cos sim {embed_type.lower()}'] = df_embed[f'cosine_pred_{embed_type}']
    return df
def main():


    # uses average of cos sim per participat
    # average per participant
    df = analyze_combined_regions(print_results = False,makebold = False)
    plt = make_boxplot(df)
    save_boxplot(plt)
    plt.show()

    # raw participant scores
    # df = get_27_participants_df()
    # plt = make_boxplot_27_participants(df)
    # save_boxplot(plt,filename='boxplot_27_participants_raw.png')
    # plt.show()


    print('Done')
if __name__ == '__main__':
    main()

