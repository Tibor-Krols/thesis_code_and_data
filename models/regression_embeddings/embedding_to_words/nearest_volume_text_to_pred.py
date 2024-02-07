import numpy as np
import pandas as pd

from dataset_loader.dataset import LPPDataset
from dataset_loader.section_participant_base import BaseSectionParticipant
from utils.embeddings import load_embeddings
from utils.paths import data_path
from sklearn.metrics.pairwise import cosine_similarity


def load_embeddings_section(dataset,embed_type,participant,section):
    i = dataset.get_participant_section_index(participant=participant,section=section)
    ps = BaseSectionParticipant(dataset[i],include_volume_words_dict=True,embed_type=embed_type)

    n_volumes = ps.nr_fmri_frames

    # get words per volume
    words = [ps.get_words_volume_idx(i) for i in range(n_volumes)]
    words_list = [
            w for w in words if w !=[]
        ]
    # select indices that contain words
    indices_vols_with_words = [
        i for i in range(n_volumes) if words[i] != []
    ]
    embeddings = np.array([ps.get_mean_embed_volume_idx(i) for i in indices_vols_with_words])
    return embeddings,words_list


def get_best_and_worst_predictions(df,embed_type):
    df = df.sort_values(by=f'cosine_pred_{embed_type}',ascending=False)
    df['gt_text'] = df[f'volume_words'].apply(lambda x: ' '.join(x))
    df[f'pred_text_{embed_type}'] = df[f'pred_words'].apply(lambda x: ' '.join(x))
    df = df[['gt_text',f'pred_text_{embed_type}',f'cosine_pred_{embed_type}']]
    df = df.rename(columns={f'cosine_pred_{embed_type}':'cos sim'})
    dfnew = pd.concat([df.head(10),df.tail(10)])


    latex_string = dfnew.to_latex(index=False)
    latex_string = latex_string.replace('_','\\_')


    print(latex_string)





def main():
    # embed_type = 'GloVe'
    embed_types  = ['GloVe', 'BERT']
    participant = 'sub-EN063'

    #participants with highest average cosine similarity
    participant_dict = {'GloVe':'sub-EN063','BERT':'sub-EN081'}
    section = 8
    for embed_type in embed_types:
    # load embeddings ground truth.
        participant = participant_dict[embed_type]
        dataset = LPPDataset(embed_type=embed_type)
        gt_embed,gt_words = load_embeddings_section(dataset=dataset,
                                                    embed_type=embed_type,
                                                    participant=participant,
                                                    section=section)

        # load embeddings predictions
        filename = f'pred_embed_{embed_type}_{participant}_section_8_Angular Gyrus_and_more.pkl'
        df_pred = pd.read_pickle(data_path / 'predictions' / embed_type / filename)
        pred_emb = np.vstack(df_pred[f'pred_{embed_type}_test'].values)
        # add row of zeroes, to diferentiate shape
        # pred_emb = np.vstack((pred_emb,np.zeros(pred_emb.shape[1])))

        # select nearest text
        cosine_similarities = cosine_similarity(gt_embed, pred_emb)
        most_similar_index = np.argmax(cosine_similarities, axis=0)
        pred_words = [gt_words[i] for i in most_similar_index]
        df_pred['pred_words'] = pred_words


        print(f'for {participant}, and {embed_type}')
        get_best_and_worst_predictions(df=df_pred, embed_type=embed_type)


        # top N
        # Iterate over each predicted embedding
        df_pred['gt_text'] = df_pred[f'volume_words'].apply(lambda x: ' '.join(x))

        N=5
        most_similar_words_list = []
        for i in range(len(pred_emb)):
            # Sort cosine similarities and get top N indices
            top_indices = np.argsort(cosine_similarities[:, i])[::-1][:N]
            # Retrieve the corresponding words from the volume_words column
            most_similar_words = df_pred.iloc[top_indices]['volume_words'].tolist()
            most_similar_words = [' '.join(l) for l in most_similar_words]
            most_similar_words_list.append(most_similar_words)
            # print(f"Predicted embedding {i + 1}: Most similar words: {most_similar_words}")
        df_pred[f'top_{N}_similar'] = most_similar_words_list
        df_pred[f'is_in_top_{N}'] = df_pred.apply(lambda row: row['gt_text'] in row[f'top_{N}_similar'], axis=1)

    dfsame = df_pred[df_pred['pred_words']==df_pred['volume_words']]
    print('done')
if __name__ == '__main__':
    main()