import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict

from dataset_loader.dataset import LPPDataset
from dataset_loader.section_participant_base import BaseSectionParticipant
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# from external_repos.nn_decoding.src import util
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
from utils.cortical_masking import get_oxford_mask, mask_timeseries
from utils.paths import *

def convert_to_array(s):
    # Remove brackets, replace '\n' with space, split by space, and convert to float
    return np.array([float(x) for x in s.strip('[]').replace('\n', ' ').split()])


def load_baseline_volume_embeddings(embed_type:str):
    filepath = pred_path/'baseline'/'embeddings'/ f'baseline_{embed_type}_embeddings_per_volume.pkl'
    df = pd.read_pickle(filepath)
    return df

def load_baseline_word_embeddings(embed_type:str):
    filepath = pred_path/'baseline'/'embeddings'/ f'baseline_{embed_type}_embeddings_per_word.pkl'
    df = pd.read_pickle(filepath)
    return df

def get_word_avg_baseline_encodings(section:int,data,indices_vols_with_words,embed_type):
    df_word_sect = data[data['sections']==section]
    result = df_word_sect.groupby('volume_idx')[embed_type].apply(lambda x: np.mean(x.tolist(), axis=0)).reset_index()
    result = result.iloc[indices_vols_with_words]
    return np.array(result[embed_type].to_list())

def get_section_baseline_encodings(section:int,data:pd.DataFrame,indices_vols_with_words,embed_type:str):
    df_sect = data[data['sections']==section]
    df_sect = df_sect.iloc[indices_vols_with_words]
    return np.array(df_sect[embed_type].to_list())
def main():
    embed_type = 'BERT'
    dataset = LPPDataset(embed_type=embed_type)
    participant = 'sub-EN057'
    section = 1
    cortex_regions = ['Superior Temporal Gyrus, anterior division']

    # TODO: get images (masked) and embeddings for multiple participant sections.

    ps_index = dataset.get_participant_section_index(participant=participant,section=section)
    ps = BaseSectionParticipant(dataset[ps_index],embed_type=embed_type)
    section = ps.section
    n_volumes = ps.nr_fmri_frames
    # n_volumes = 15 #small for testing purposes

    # get words per volume
    words = [ps.get_words_volume_idx(i) for i in range(n_volumes)]
    # select indices that contain words
    indices_vols_with_words = [
        i for i in range(n_volumes) if words[i] != []
    ]

    # load baseline
    # Baseline_section = get_section_baseline_encodings(
    #     section=section,
    #     data=load_baseline_volume_embeddings(embed_type=embed_type),
    #     indices_vols_with_words=indices_vols_with_words,
    #     embed_type=embed_type
    # )
    Baseline_section = get_word_avg_baseline_encodings(
        section=section,
        data=load_baseline_volume_embeddings(embed_type=embed_type),
        indices_vols_with_words=indices_vols_with_words,
        embed_type=embed_type
    )
    # get images
    subject_images = ps.fmri[...,indices_vols_with_words].numpy()

    # mask_images
    # load mask
    cortical_mask = get_oxford_mask(cortical_regions= cortex_regions)
    # mask_volumes
    subject_images = mask_timeseries(timeseries= subject_images,cortical_mask=cortical_mask)

    # load embeddings
    # # get embeddings of volumes: calculate mean if multiple words.
    encodings = np.array([ps.get_mean_embed_volume_idx(i) for i in indices_vols_with_words])

    # Final data prep: normalize.
    X = subject_images - subject_images.mean(axis=0)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    # X=np.nan_to_num(X) #fill na (not needed now
    Y = encodings - encodings.mean(axis=0)
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    Baseline_section = Baseline_section - Baseline_section.mean(axis=0)
    Baseline_section = Baseline_section / np.linalg.norm(Baseline_section, axis=1, keepdims=True)




    # Prepare nested CV.
    # Inner CV is responsible for hyperparameter optimization;
    # # outer CV is responsible for prediction.
    state = 42
    n_folds = 3
    ALPHAS = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e1]
    inner_cv = KFold(n_splits=n_folds, shuffle=True, random_state=state)
    outer_cv = KFold(n_splits=n_folds, shuffle=True, random_state=state)

    # Run inner CV.
    gs = GridSearchCV(Ridge(fit_intercept=False,),
                      {"alpha": ALPHAS}, cv=inner_cv, verbose=10)
    # Run outer CV.
    decoder_predictions = cross_val_predict(gs, X, Y, cv=outer_cv)



    # new attempt ridge regression:
    # Reshape the fMRI data to 2D (flattening the spatial dimensions)
    if X[...,0].shape == (73, 90, 74):
        X = X.reshape((X.shape[0] * X.shape[1] * X.shape[2],X.shape[3])).T

    # select indices with words from x
    # X = X[indices_vols_with_words,]
    assert len(encodings)== len(X)

    # Reshape the labels to 2D
    # Y_reshaped = Y

    # Split the data into training and testing sets
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_test, Y_train, Y_test, Baseline_train, Baseline_test = train_test_split(X, Y, Baseline_section, test_size=0.2, random_state=42)

    # Fit a Ridge Regression model
    alpha = 1.0  # You can adjust this regularization parameter
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train, Y_train)

    # Predict on the test set
    Y_pred = ridge_model.predict(X_test)

    # Evaluate the model (you can use different metrics depending on your task)
    mse = mean_squared_error(Y_test, Y_pred)
    print(f'Mean Squared Error: {mse}')
    mse = mean_squared_error(Y_test, Y_pred)

    # cosine similarity
    assert Y_test.shape == Y_pred.shape
    cosine_similarities = cosine_similarity(Y_pred, Y_test)
    cosine_similarities = np.diagonal(cosine_similarities, axis1=0, axis2=1)
    print(cosine_similarities.mean())

    # cosine similarity with baseline
    cosine_similarities = cosine_similarity(Baseline_test, Y_test)
    cosine_similarities = np.diagonal(cosine_similarities, axis1=0, axis2=1)
    print('Baseline ',cosine_similarities.mean())

    # pred on train data
    Y_pred_train = ridge_model.predict(X_train)
    cosine_similarities_train = cosine_similarity(Y_pred_train, Y_train)
    cosine_similarities_train = np.diagonal(cosine_similarities_train, axis1=0, axis2=1)
    print(cosine_similarities_train.mean())


    #evaluate cross val predictions
    # decoder_predictions
    # Y_pred_train = ridge_model.predict(X_train)
    cosine_similarities_cross_val = cosine_similarity(decoder_predictions, Y)
    cosine_similarities_cross_val = np.diagonal(cosine_similarities_cross_val, axis1=0, axis2=1)
    print(cosine_similarities_cross_val.mean())

    print('end')


if __name__ == '__main__':
    main()