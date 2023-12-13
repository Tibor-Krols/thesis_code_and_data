import pandas as pd
# import torch
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

def load_volumes_embeddings_baseline(
        dataset:LPPDataset,
        participant:str,
        cortex_regions:list[str],
        embed_type:str,
        sections:list[int]=range(1,10)
):
    # initialize variables
    encodings_total = None
    subject_images_total = None
    baseline_total = None
    for section in sections:
        ps_index = dataset.get_participant_section_index(participant=participant,section=section)
        ps = BaseSectionParticipant(dataset[ps_index],embed_type=embed_type)
        # section = ps.section
        n_volumes = ps.nr_fmri_frames
        n_volumes = 15 #small for testing purposes

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
        # TODO: get most informative voxels like pereira et al.
        # load mask
        cortical_mask = get_oxford_mask(cortical_regions= cortex_regions)
        # mask_volumes
        subject_images = mask_timeseries(timeseries= subject_images,cortical_mask=cortical_mask)

        # load embeddings
        # # get embeddings of volumes: calculate mean if multiple words.
        encodings = np.array([ps.get_mean_embed_volume_idx(i) for i in indices_vols_with_words])

        #concatenate section info to total
        #encodings
        if encodings_total is None:
            encodings_total = encodings
        else:
            encodings_total = np.concatenate([encodings_total,encodings], axis = 0)
        #images
        if subject_images_total is None:
            subject_images_total = subject_images
        else:
            subject_images_total = np.concatenate([subject_images_total,subject_images])
        #baseline
        if baseline_total is None:
            baseline_total = Baseline_section
        else:
            baseline_total = np.concatenate([baseline_total,Baseline_section], axis = 0)

    # Final data prep: normalize.
    X = subject_images_total - subject_images_total.mean(axis=0)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    # X=np.nan_to_num(X) #fill na (not needed now
    Y = encodings_total - encodings_total.mean(axis=0)
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    baseline_total = baseline_total - baseline_total.mean(axis=0)
    baseline_total = baseline_total / np.linalg.norm(baseline_total, axis=1, keepdims=True)

    return X,Y,baseline_total


def main():
    embed_type = 'BERT'
    dataset = LPPDataset(embed_type=embed_type)
    participant = 'sub-EN057'
    section = 1
    cortex_regions = ['Superior Temporal Gyrus, anterior division']
    # TODO: optimize alpha on train set, make predictions on test set
    # 7/9 sections train: 2/9 sections test?
    X,Y,Y_baseline = load_volumes_embeddings_baseline(
        dataset=dataset,
        participant=participant,
        cortex_regions=cortex_regions,
        embed_type=embed_type,
        sections=range(1,10)
    )
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
    gs.fit(X,Y)
    best_alpha = gs.best_params_['alpha']
    # Run outer CV.
    # decoder_predictions = cross_val_predict(gs, X, Y, cv=outer_cv)



    # new attempt ridge regression:
    # Reshape the fMRI data to 2D (flattening the spatial dimensions)
    if X[...,0].shape == (73, 90, 74):
        X = X.reshape((X.shape[0] * X.shape[1] * X.shape[2],X.shape[3])).T

    # select indices with words from x
    # X = X[indices_vols_with_words,]
    assert len(Y) == len(X)

    # Split the data into training and testing sets
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_test, Y_train, Y_test, Baseline_train, Baseline_test = train_test_split(X, Y, Y_baseline, test_size=0.2, random_state=42)

    # Fit a Ridge Regression model
    alpha = best_alpha  # You can adjust this regularization parameter
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
    # # Y_pred_train = ridge_model.predict(X_train)
    # cosine_similarities_cross_val = cosine_similarity(decoder_predictions, Y)
    # cosine_similarities_cross_val = np.diagonal(cosine_similarities_cross_val, axis1=0, axis2=1)
    # print(cosine_similarities_cross_val.mean())

    print('end')


if __name__ == '__main__':
    main()