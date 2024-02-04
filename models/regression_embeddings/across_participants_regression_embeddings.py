import os
import random
import pandas as pd
# import torch
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from tqdm import tqdm

from dataset_loader.dataset import LPPDataset
from dataset_loader.section_participant_base import BaseSectionParticipant
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# from external_repos.nn_decoding.src import util
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from models.regression_embeddings.regression_embeddings import get_word_avg_baseline_encodings, \
    load_baseline_word_embeddings
from models.regression_embeddings.voxel_selection.voxel_selection_text import load_selected_voxel_mask
from utils.cortical_masking import mask_timeseries, get_aal_mask, get_oxford_mask, show_mask, load_mask_both_atlasses, \
    check_atlas_type
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from dataset_loader.dataset import LPPDataset
from dataset_loader.section_participant_base import BaseSectionParticipant
import numpy as np
from models.regression_embeddings.voxel_selection.voxel_selection_text import load_selected_voxel_mask
from utils.cortical_masking import mask_timeseries, get_aal_mask, get_oxford_mask, show_mask, load_mask_both_atlasses
from utils.paths import *


def load_volumes_embeddings_multiple_participants(
        dataset:LPPDataset,
        participants:list[str],
        embed_type:str,
        sections:list[int]=range(1,10),
        cortex_regions: list[str]=None,

):
    """if cortex_regions defined, uses selected regions. otherwise uses voxel selection process """
    # initialize variables
    encodings_total = None
    subject_images_total = None
    baseline_total = None
    section_volume_idx_dict = {}
    section_words_dict = {}

    # load mask
    if cortex_regions is not None:
        # use regions if specified
        # load from relevant atlas
        atlas_type = check_atlas_type(cortex_regions)
        if atlas_type == 'both':
            cortical_mask = load_mask_both_atlasses(cortical_regions=cortex_regions)
        elif atlas_type == 'harvard_oxford':
            cortical_mask = get_oxford_mask(cortical_regions=cortex_regions)
        elif atlas_type == 'aal':
            cortical_mask = get_aal_mask(cortical_regions=cortex_regions)

    else:
        # otherwise use selected voxels
        # load informative voxel mask (like pereira et al.(2018))
        cortical_mask = load_selected_voxel_mask(
            participant=participants,
            embed_type=embed_type,
            filepath=data_path / 'voxel_selection_masks',
            selection_criterium='avg'
            # TODO: if use selection criterium. decide and specify which to use
        )

    for participant in participants:

        # loop over sections
        for section in tqdm(sections):
            ps_index = dataset.get_participant_section_index(participant=participant, section=section)
            ps = BaseSectionParticipant(dataset[ps_index],embed_type=embed_type)
            # section = ps.section
            n_volumes = ps.nr_fmri_frames
            # TODO: use all volumes
            # n_volumes = 5 #small for testing purposes

            # get words per volume
            words = [ps.get_words_volume_idx(i) for i in range(n_volumes)]
            # select indices that contain words
            indices_vols_with_words = [
                i for i in range(n_volumes) if words[i] != []
            ]
            words_list = [
                w for w in words if w !=[]
            ]
            section_words_dict[section] = words_list
            section_volume_idx_dict[section] = indices_vols_with_words
            # load baseline
            # Baseline_section = get_section_baseline_encodings(
            #     section=section,
            #     data=load_baseline_volume_embeddings(embed_type=embed_type),
            #     indices_vols_with_words=indices_vols_with_words,
            #     embed_type=embed_type
            # )
            Baseline_section = get_word_avg_baseline_encodings(
                section=section,
                data=load_baseline_word_embeddings(embed_type=embed_type),
                indices_vols_with_words=indices_vols_with_words,
                embed_type=embed_type
            )
            # get images
            subject_images = ps.fmri[...,indices_vols_with_words].numpy()

            # mask_images
            # mask_volumes
            subject_images = mask_timeseries(timeseries = subject_images,cortical_mask = cortical_mask)

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

    return X,Y,baseline_total,section_words_dict,section_volume_idx_dict




def train_ridge_across_participants(
        embed_type: str,
        dataset,
        train_participants: list[str],
        cortex_regions: list[str],
        train_sections: list[int],
):
    # load train data
    (X_train, Y_train,
     Y_baseline_train,
     section_words_dict_train,
     section_volume_idx_dict_train) = load_volumes_embeddings_multiple_participants(
        dataset=dataset,
        participants=train_participants,
        embed_type=embed_type,
        sections=train_sections,
        cortex_regions=cortex_regions
    )

    # Prepare nested CV.
    # Inner CV is responsible for hyperparameter optimization;
    # # outer CV is responsible for prediction.
    state = 42
    n_folds = 5
    ALPHAS = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e1]
    inner_cv = KFold(n_splits=n_folds, shuffle=True, random_state=state)
    outer_cv = KFold(n_splits=n_folds, shuffle=True, random_state=state)

    # Run inner CV.
    gs = GridSearchCV(Ridge(fit_intercept=False, ),
                      {"alpha": ALPHAS}, cv=inner_cv, verbose=10)
    gs.fit(X_train, Y_train)
    best_alpha = gs.best_params_['alpha']

    # Reshape the fMRI data to 2D (flattening the spatial dimensions)
    if X_train[...,0].shape == (73, 90, 74):
        X_train = X_train.reshape((X_train.shape[0] * X_train.shape[1] * X_train.shape[2],X_train.shape[3])).T

    assert len(Y_train) == len(X_train)

    # Fit a Ridge Regression model
    alpha = best_alpha  # You can adjust this regularization parameter
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train, Y_train)
    return ridge_model, best_alpha

def test_ridge_model(
        embed_type: str,
        dataset,
        test_participants: list[str],
        test_sections: list[int],
        cortex_regions: list[str],
        ridge_model
):
    # load test data
    (X_test, Y_test,
     Y_baseline_test,
     section_words_dict_test,
     section_volume_idx_dict_test) = load_volumes_embeddings_multiple_participants(
        dataset=dataset,
        participants=test_participants,
        embed_type=embed_type,
        sections=test_sections,
        cortex_regions=cortex_regions
    )

    # format test data
    if X_test[..., 0].shape == (73, 90, 74):
        X_test = X_test.reshape((X_test.shape[0] * X_test.shape[1] * X_test.shape[2], X_test.shape[3])).T
    assert len(Y_test) == len(X_test)

    # make predictions
    # Predict on the test set
    Y_pred = ridge_model.predict(X_test)
    return Y_pred



def run_ridge_across_participants(
        embed_type:str,
        dataset,
        train_participants:list[str],
        test_participants:list[str],
        cortex_regions:list[str],
        train_sections:list[int],
        test_sections:list[int]
):
    nr_test_participants = len(test_participants)
    nr_test_sections = len(test_sections)
    # train model and hyper parameter tune
    ridge_model, best_alpha = train_ridge_across_participants(
        embed_type=embed_type,
        dataset=dataset,
        train_participants=train_participants,
        cortex_regions=cortex_regions,
        train_sections=train_sections
    )

    # test model
    # Y_pred = test_ridge_model(
    #     embed_type=embed_type,
    #     dataset=dataset,
    #     test_participants=test_participants,
    #     test_sections=test_sections,
    #     cortex_regions=cortex_regions,
    #     ridge_model=fit_ridge_model
    # )
    # load test data
    (X_test, Y_test,
     Y_baseline_test,
     section_words_dict_test,
     section_volume_idx_dict_test) = load_volumes_embeddings_multiple_participants(
        dataset=dataset,
        participants=test_participants,
        embed_type=embed_type,
        sections=test_sections,
        cortex_regions=cortex_regions
    )

    # format test data
    if X_test[..., 0].shape == (73, 90, 74):
        X_test = X_test.reshape((X_test.shape[0] * X_test.shape[1] * X_test.shape[2], X_test.shape[3])).T
    assert len(Y_test) == len(X_test)

    # make predictions
    # Predict on the test set
    Y_pred = ridge_model.predict(X_test)


    #evaluate_predictions
    # Evaluate the model (you can use different metrics depending on your task)
    mse = mean_squared_error(Y_test, Y_pred)
    print(f'Mean Squared Error: {mse}')
    mse = mean_squared_error(Y_test, Y_pred)

    # cosine similarity
    assert Y_test.shape == Y_pred.shape
    cosine_similarity_test = cosine_similarity(Y_pred, Y_test)
    cosine_similarity_test = np.diagonal(cosine_similarity_test, axis1=0, axis2=1)
    print(f'cosine similarity test pred: {cosine_similarity_test.mean()}')

    # cosine similarity with baseline
    cosine_similarities_base = cosine_similarity(Y_baseline_test, Y_test)
    cosine_similarities_base = np.diagonal(cosine_similarities_base, axis1=0, axis2=1)
    print('Baseline cosine similarity: ', cosine_similarities_base.mean())

    # save predictions and scores
    # nr of volumes
    nr_of_test_volumes = sum([len(l) for l in section_volume_idx_dict_test.values()])
    nr_of_test_volumes = len(section_volume_idx_dict_test[test_sections[0]])
    df = pd.DataFrame({
        'participant':[p for p in test_participants
                       for nr_of_test_volumes in [len(l) for l in section_volume_idx_dict_test.values()]
                       for i in range(nr_of_test_volumes)
                       ],
        'section': [sec for sec, words in section_words_dict_test.items() for _ in words] * nr_test_participants,
        f'volume_words': [w for words in section_words_dict_test.values() for w in words] * nr_test_participants,
        'volume_idx': [i for indices in section_volume_idx_dict_test.values() for i in indices] * nr_test_participants,
        f'pred_{embed_type}_test': Y_pred.tolist(),
        f'baseline_{embed_type}_test': Y_baseline_test.tolist(),
        f'cosine_pred_{embed_type}': cosine_similarity_test,
        f'cosine_baseline_{embed_type}': cosine_similarities_base,
        'alpha': best_alpha
    })

    # add unseen participant unseen section
    df['unseen_participant'] = [p not in train_participants for p in df.participant]
    df['unseen_section'] = [p not in train_sections for p in df.section]
    df['fully_unseen'] = df['unseen_participant'] & df['unseen_section']

    # save file
    train_participants_string = '_'.join(train_participants)
    train_sections_string = '_'.join([str(s) for s in train_sections])
    filepath = data_path/'predictions'/'across_participants'/embed_type/f'train_participants_{train_participants_string}_train_sections_{train_sections_string}'
    os.makedirs(filepath,exist_ok=True)


    #specify filename and folder
    test_sections_str = '_'.join([str(s) for s in test_sections])
    test_participants_str = '_'.join(test_participants)
    if cortex_regions is not None:
        region_names = '_'.join(cortex_regions[:-6])
        if len(region_names)>150:
            region_names = cortex_regions[0] + '_and_more'
        filename = f"pred_embed_{embed_type}_test_{test_participants_str}_test_sections_{test_sections_str}_{region_names}.pkl"
    else:
        filename = f"pred_embed_{embed_type}_test_{test_participants_str}_test_sections_{test_sections_str}.pkl"


    df.to_pickle(filepath/filename)
    print(f'done running ridge for {cortex_regions} for {embed_type}')




def run_between_participants_both_embeddings():

    # TODO: specify good train and test participants and sections
    train_participants = ['sub-EN077', 'sub-EN069', 'sub-EN099', 'sub-EN086', 'sub-EN104']
    train_sections = [1,3,5,8]
    test_participants = ['sub-EN086', 'sub-EN104', 'sub-EN101', 'sub-EN106']
    test_sections = [2,8]


    ####### BERT #######
    embed_type = 'BERT'
    dataset = LPPDataset(embed_type=embed_type)

    # Regions for bert
    cortex_regions = [
            'Angular Gyrus',
            'Cerebelum_Crus1_L', 'Cerebelum_Crus1_R', 'Cerebelum_Crus2_L', 'Cerebelum_Crus2_R', 'Cerebelum_3_L',
            'Cerebelum_3_R', 'Cerebelum_4_5_L', 'Cerebelum_4_5_R', 'Cerebelum_6_L', 'Cerebelum_6_R', 'Cerebelum_7b_L',
            'Cerebelum_7b_R', 'Cerebelum_8_L', 'Cerebelum_8_R', 'Cerebelum_9_L', 'Cerebelum_9_R', 'Cerebelum_10_L',
            'Cerebelum_10_R',
            'Superior Temporal Gyrus, anterior division', 'Superior Temporal Gyrus, posterior division',
            'Temporal_Sup_L', 'Temporal_Pole_Sup_L', 'Temporal_Mid_L', 'Temporal_Pole_Mid_L', 'Temporal_Inf_L',
            'Temporal_Sup_R', 'Temporal_Pole_Sup_R', 'Temporal_Mid_R', 'Temporal_Pole_Mid_R', 'Temporal_Inf_R'
        ]

    dataset = LPPDataset(embed_type=embed_type)
    run_ridge_across_participants(
        embed_type=embed_type,
        dataset=dataset,
        train_participants=train_participants,
        test_participants=test_participants,
        cortex_regions=cortex_regions,
        train_sections=train_sections,
        test_sections=test_sections
    )


    ############ GloVe ##########
    embed_type = 'GloVe'
    cortex_regions = [
            'Angular Gyrus',
            'Cerebelum_Crus1_L', 'Cerebelum_Crus1_R', 'Cerebelum_Crus2_L', 'Cerebelum_Crus2_R', 'Cerebelum_3_L',
            'Cerebelum_3_R', 'Cerebelum_4_5_L', 'Cerebelum_4_5_R', 'Cerebelum_6_L', 'Cerebelum_6_R', 'Cerebelum_7b_L',
            'Cerebelum_7b_R', 'Cerebelum_8_L', 'Cerebelum_8_R', 'Cerebelum_9_L', 'Cerebelum_9_R', 'Cerebelum_10_L',
            'Cerebelum_10_R',
            'Superior Temporal Gyrus, anterior division', 'Superior Temporal Gyrus, posterior division',
            'Temporal_Sup_R', 'Temporal_Pole_Sup_R', 'Temporal_Mid_R', 'Temporal_Pole_Mid_R', 'Temporal_Inf_R'
        ]

    dataset = LPPDataset(embed_type=embed_type)
    run_ridge_across_participants(
        embed_type=embed_type,
        dataset=dataset,
        train_participants=train_participants,
        test_participants=test_participants,
        cortex_regions=cortex_regions,
        train_sections=train_sections,
        test_sections=test_sections
    )


def main():
    run_between_participants_both_embeddings()
    embed_types = ['GloVe','BERT']
    train_participants = ['sub-EN057','sub-EN058']
    train_sections = [1,5]
    test_participants = ['sub-EN058','sub-EN059','sub-EN061']
    test_sections = [5,8]
    cortex_regions = ['Angular Gyrus']

    for embed_type in embed_types:
        dataset = LPPDataset(embed_type=embed_type)
        run_ridge_across_participants(
            embed_type=embed_type,
            dataset=dataset,
            train_participants=train_participants,
            test_participants=test_participants,
            cortex_regions=cortex_regions,
            train_sections=train_sections,
            test_sections=test_sections
        )
    print('done')

if __name__ == '__main__':
    main()