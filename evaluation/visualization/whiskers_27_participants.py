
import pandas as pd
import matplotlib.pyplot as plt

from evaluation.visualization.boxplot_plot_many_participants import save_boxplot


def plot_errorbar(df,embed_type):
    baseline_GloVe = round(0.002532702616162057,6)
    baseline_BERT = round(-0.0035220872142616455,6)
    baseline_GloVe_std =round( 0.059781804096505346,6)
    baseline_BERT_std = round(0.03702856899118565,6)
    if embed_type=="GloVe":
        baseline = baseline_GloVe
    if embed_type == 'BERT':
        baseline= baseline_BERT

    # Plot
    # horizontal
    plt.figure(figsize=(9,8))
    plt.errorbar(df['cosine_similarity'], df['participant'], xerr=df['cosine_similarity_std'], fmt='o', capsize=5, label='Mean ± Std')
    plt.scatter(df['cosine_similarity'], df['participant'], color='red')  # Plot mean as dots
    plt.axvline(x=baseline, color='red', linestyle='--', label='Baseline')

    # vertical
    # Plot mean as dots
    # plt.figure(figsize=(8, 10))
    # plt.errorbar(df['participant'], df['cosine_similarity'], yerr=df['cosine_similarity_std'], fmt='o', capsize=5, label='Mean ± Std')
    # plt.scatter(df['participant'], df['cosine_similarity'], color='red')
    # plt.axhline(y=baseline, color='red', linestyle='--', label='Baseline')
    # plt.xticks(rotation=90)


    plt.ylabel('Participant',fontsize = 'xx-large')
    plt.xlabel('Cosine Similarity with Ground Truth',fontsize = 'xx-large')
    plt.title(f'{embed_type}\n\n',fontsize = 'xx-large')
    # plt.xlim(right= 0.4)
    # plt.legend(fontsize = 'x-large')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1.18),fontsize= 'x-large')

    # plt.tight_layout(rect=[0, 0, 1, 0.70])
    plt.tight_layout()

    # plt.grid(True)
    # save plot
    save_boxplot(plt,filename=f'whiskers_27_part_{embed_type}.png')

    plt.show()
def main():
    # load data
    df = pd.read_csv(r'F:\dataset\predictions\overall\overview_predictions.csv')

    # round decimals
    df = df.round(decimals=6)


    bert_regions = 'Angular Gyrus_Cerebelum_Crus1_L_Cerebelum_Crus1_R_Cerebelum_Crus2_L_Cerebelum_Crus2_R_Cerebelum_3_L_Cerebelum_3_R_Cerebelum_4_5_L_Cerebelum_4_5_R_Cerebelum_6_L_Cerebelum_6_R_Cerebelum_7b_L_Cerebelum_7b_R_Cerebelum_8_L_Cerebelum_8_R_Cerebelum_9_L_Cerebelum_9_R_Cerebelum_10_L_Cerebelum_10_R_Superior Temporal Gyrus, anterior division_Superior Temporal Gyrus, posterior division_Temporal_Sup_L_Temporal_Pole_Sup_L_Temporal_Mid_L_Temporal_Pole_Mid_L_Temporal_Inf_L_Temporal_Sup_R_Temporal_Pole_Sup_R_Temporal_Mid_R_Temporal_Pole_Mid_R_Temporal_Inf_R'
    glove_regions = 'Angular Gyrus_Cerebelum_Crus1_L_Cerebelum_Crus1_R_Cerebelum_Crus2_L_Cerebelum_Crus2_R_Cerebelum_3_L_Cerebelum_3_R_Cerebelum_4_5_L_Cerebelum_4_5_R_Cerebelum_6_L_Cerebelum_6_R_Cerebelum_7b_L_Cerebelum_7b_R_Cerebelum_8_L_Cerebelum_8_R_Cerebelum_9_L_Cerebelum_9_R_Cerebelum_10_L_Cerebelum_10_R_Superior Temporal Gyrus, anterior division_Superior Temporal Gyrus, posterior division_Temporal_Sup_R_Temporal_Pole_Sup_R_Temporal_Mid_R_Temporal_Pole_Mid_R_Temporal_Inf_R'

    dfbert = df[df.embed_type=="BERT"]
    dfglove = df[df.embed_type=='GloVe']

    dfbert = dfbert[dfbert.cortex_regions == bert_regions]
    dfbert.drop_duplicates(inplace=True)
    dfglove = dfglove[dfglove.cortex_regions == glove_regions]
    dfglove.drop_duplicates(inplace=True)

    # sort values by participant
    dfglove = dfglove.sort_values(by='participant',ascending =False)
    dfbert = dfbert.sort_values(by='participant',ascending =False)


    plot_errorbar(dfglove,embed_type='GloVe')

    plot_errorbar(dfbert,embed_type='BERT')


    print('done')

if __name__ == '__main__':
    main()