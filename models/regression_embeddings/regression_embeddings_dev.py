import os

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

from models.regression_embeddings.voxel_selection.voxel_selection_text import load_selected_voxel_mask
from utils.cortical_masking import mask_timeseries, get_aal_mask,get_oxford_mask
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

def check_atlas_type(cortical_regions)->str:
    # TODO: remove hardcoding and load labels
    # currently hardcoded to speed up runs and not load labels every time
    labels_harvard_oxford = ['Background', 'Frontal Pole', 'Insular Cortex', 'Superior Frontal Gyrus', 'Middle Frontal Gyrus', 'Inferior Frontal Gyrus, pars triangularis', 'Inferior Frontal Gyrus, pars opercularis', 'Precentral Gyrus', 'Temporal Pole', 'Superior Temporal Gyrus, anterior division', 'Superior Temporal Gyrus, posterior division', 'Middle Temporal Gyrus, anterior division', 'Middle Temporal Gyrus, posterior division', 'Middle Temporal Gyrus, temporooccipital part', 'Inferior Temporal Gyrus, anterior division', 'Inferior Temporal Gyrus, posterior division', 'Inferior Temporal Gyrus, temporooccipital part', 'Postcentral Gyrus', 'Superior Parietal Lobule', 'Supramarginal Gyrus, anterior division', 'Supramarginal Gyrus, posterior division', 'Angular Gyrus', 'Lateral Occipital Cortex, superior division', 'Lateral Occipital Cortex, inferior division', 'Intracalcarine Cortex', 'Frontal Medial Cortex', 'Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)', 'Subcallosal Cortex', 'Paracingulate Gyrus', 'Cingulate Gyrus, anterior division', 'Cingulate Gyrus, posterior division', 'Precuneous Cortex', 'Cuneal Cortex', 'Frontal Orbital Cortex', 'Parahippocampal Gyrus, anterior division', 'Parahippocampal Gyrus, posterior division', 'Lingual Gyrus', 'Temporal Fusiform Cortex, anterior division', 'Temporal Fusiform Cortex, posterior division', 'Temporal Occipital Fusiform Cortex', 'Occipital Fusiform Gyrus', 'Frontal Operculum Cortex', 'Central Opercular Cortex', 'Parietal Operculum Cortex', 'Planum Polare', "Heschl's Gyrus (includes H1 and H2)", 'Planum Temporale', 'Supracalcarine Cortex', 'Occipital Pole']
    labels_aal = ['Precentral_L', 'Precentral_R', 'Frontal_Sup_L', 'Frontal_Sup_R', 'Frontal_Sup_Orb_L', 'Frontal_Sup_Orb_R', 'Frontal_Mid_L', 'Frontal_Mid_R', 'Frontal_Mid_Orb_L', 'Frontal_Mid_Orb_R', 'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R', 'Frontal_Inf_Tri_L', 'Frontal_Inf_Tri_R', 'Frontal_Inf_Orb_L', 'Frontal_Inf_Orb_R', 'Rolandic_Oper_L', 'Rolandic_Oper_R', 'Supp_Motor_Area_L', 'Supp_Motor_Area_R', 'Olfactory_L', 'Olfactory_R', 'Frontal_Sup_Medial_L', 'Frontal_Sup_Medial_R', 'Frontal_Med_Orb_L', 'Frontal_Med_Orb_R', 'Rectus_L', 'Rectus_R', 'Insula_L', 'Insula_R', 'Cingulum_Ant_L', 'Cingulum_Ant_R', 'Cingulum_Mid_L', 'Cingulum_Mid_R', 'Cingulum_Post_L', 'Cingulum_Post_R', 'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L', 'ParaHippocampal_R', 'Amygdala_L', 'Amygdala_R', 'Calcarine_L', 'Calcarine_R', 'Cuneus_L', 'Cuneus_R', 'Lingual_L', 'Lingual_R', 'Occipital_Sup_L', 'Occipital_Sup_R', 'Occipital_Mid_L', 'Occipital_Mid_R', 'Occipital_Inf_L', 'Occipital_Inf_R', 'Fusiform_L', 'Fusiform_R', 'Postcentral_L', 'Postcentral_R', 'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L', 'Parietal_Inf_R', 'SupraMarginal_L', 'SupraMarginal_R', 'Angular_L', 'Angular_R', 'Precuneus_L', 'Precuneus_R', 'Paracentral_Lobule_L', 'Paracentral_Lobule_R', 'Caudate_L', 'Caudate_R', 'Putamen_L', 'Putamen_R', 'Pallidum_L', 'Pallidum_R', 'Thalamus_L', 'Thalamus_R', 'Heschl_L', 'Heschl_R', 'Temporal_Sup_L', 'Temporal_Sup_R', 'Temporal_Pole_Sup_L', 'Temporal_Pole_Sup_R', 'Temporal_Mid_L', 'Temporal_Mid_R', 'Temporal_Pole_Mid_L', 'Temporal_Pole_Mid_R', 'Temporal_Inf_L', 'Temporal_Inf_R', 'Cerebelum_Crus1_L', 'Cerebelum_Crus1_R', 'Cerebelum_Crus2_L', 'Cerebelum_Crus2_R', 'Cerebelum_3_L', 'Cerebelum_3_R', 'Cerebelum_4_5_L', 'Cerebelum_4_5_R', 'Cerebelum_6_L', 'Cerebelum_6_R', 'Cerebelum_7b_L', 'Cerebelum_7b_R', 'Cerebelum_8_L', 'Cerebelum_8_R', 'Cerebelum_9_L', 'Cerebelum_9_R', 'Cerebelum_10_L', 'Cerebelum_10_R', 'Vermis_1_2', 'Vermis_3', 'Vermis_4_5', 'Vermis_6', 'Vermis_7', 'Vermis_8', 'Vermis_9', 'Vermis_10']
    harvard_oxford = set(cortical_regions) & set(labels_harvard_oxford)
    aal = set(cortical_regions) & set(labels_aal)
    if len(harvard_oxford):
        return 'harvard_oxford'
    elif len(aal):
        return 'aal'

def load_volumes_embeddings_baseline(
        dataset:LPPDataset,
        participant:str,
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
        if atlas_type == 'harvard_oxford':
            cortical_mask = get_oxford_mask(cortical_regions=cortex_regions)
        elif atlas_type == 'aal':
            cortical_mask = get_aal_mask(cortical_regions=cortex_regions)
    else:
        # otherwise use selected voxels
        # load informative voxel mask (like pereira et al.(2018))
        cortical_mask = load_selected_voxel_mask(
            participant=participant,
            embed_type=embed_type,
            filepath=data_path / 'voxel_selection_masks',
            selection_criterium='avg'
            # TODO: if use selection criterium. decide and specify which to use
        )
    # loop over sections
    for section in tqdm(sections):
        ps_index = dataset.get_participant_section_index(participant=participant,section=section)
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

def run_ridge(
        embed_type:str,
        dataset,
        participant:str,
        cortex_regions:list[str],
        train_sections:list[int],
        test_sections:list[int]
) -> pd.DataFrame:
    (X_test,Y_test,
     Y_baseline_test,section_words_dict_test,
     section_volume_idx_dict_test) = load_volumes_embeddings_baseline(
        dataset=dataset,
        participant=participant,
        embed_type=embed_type,
        sections=test_sections,
        cortex_regions=cortex_regions
    )
    (X_train,Y_train,
     Y_baseline_train,
     section_words_dict_train,
     section_volume_idx_dict_train) = load_volumes_embeddings_baseline(
        dataset=dataset,
        participant=participant,
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
    gs = GridSearchCV(Ridge(fit_intercept=False,),
                      {"alpha": ALPHAS}, cv=inner_cv, verbose=10)
    gs.fit(X_train, Y_train)
    best_alpha = gs.best_params_['alpha']
    # Run outer CV.
    # decoder_predictions = cross_val_predict(gs, X, Y, cv=outer_cv)



    # new attempt ridge regression:
    # Reshape the fMRI data to 2D (flattening the spatial dimensions)
    if X_train[...,0].shape == (73, 90, 74):
        X_train = X_train.reshape((X_train.shape[0] * X_train.shape[1] * X_train.shape[2],X_train.shape[3])).T
    if X_test[...,0].shape == (73, 90, 74):
        X_test = X_test.reshape((X_test.shape[0] * X_test.shape[1] * X_test.shape[2],X_test.shape[3])).T

    # select indices with words from x
    # X = X[indices_vols_with_words,]
    assert len(Y_train) == len(X_train)
    assert len(Y_test) == len(X_test)


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
    cosine_similarity_test = cosine_similarity(Y_pred, Y_test)
    cosine_similarity_test = np.diagonal(cosine_similarity_test, axis1=0, axis2=1)
    print(f'cosine similarity test pred: {cosine_similarity_test.mean()}')

    # cosine similarity with baseline
    cosine_similarities_base = cosine_similarity(Y_baseline_test, Y_test)
    cosine_similarities_base = np.diagonal(cosine_similarities_base, axis1=0, axis2=1)
    print('Baseline cosine similarity: ',cosine_similarities_base.mean())

    # pred on train data
    Y_pred_train = ridge_model.predict(X_train)
    cosine_similarities_train = cosine_similarity(Y_pred_train, Y_train)
    cosine_similarities_train = np.diagonal(cosine_similarities_train, axis1=0, axis2=1)
    print('pred on traind data',cosine_similarities_train.mean())

    # save predictions and scores
    df = pd.DataFrame({
        'section': [sec for sec, words in section_words_dict_test.items() for _ in words],
        f'volume_words': [w for words in section_words_dict_test.values() for w in words],
        'volume_idx': [i for indices in section_volume_idx_dict_test.values() for i in indices],
        f'pred_{embed_type}_test':Y_pred.tolist(),
        f'baseline_{embed_type}_test':Y_baseline_test.tolist(),
        f'cosine_pred_{embed_type}':cosine_similarity_test,
        f'cosine_baseline_{embed_type}': cosine_similarities_base,
        'alpha': alpha
    })
    filepath = data_path/'predictions'/embed_type
    os.makedirs(filepath,exist_ok=True)
    test_sections_str = '_'.join([str(s) for s in test_sections])
    if cortex_regions is not None:
        region_names = '_'.join(cortex_regions)
        filename = f"pred_embed_{embed_type}_{participant}_section_{test_sections_str}_{region_names}.pkl"
    else:
        filename = f"pred_embed_{embed_type}_{participant}_section_{test_sections_str}.pkl"

    df.to_pickle(filepath/filename)
    return df

def main():
    embed_type = 'GloVe'
    dataset = LPPDataset(embed_type=embed_type)
    participant = 'sub-EN057'
    train_sections = [1,2,3,4,5,6,7,9]
    test_sections = [8]
    cortex_regions = ['Cerebelum_Crus1_L', 'Cerebelum_Crus1_R', 'Cerebelum_Crus2_L', 'Cerebelum_Crus2_R', 'Cerebelum_3_L', 'Cerebelum_3_R', 'Cerebelum_4_5_L', 'Cerebelum_4_5_R', 'Cerebelum_6_L', 'Cerebelum_6_R', 'Cerebelum_7b_L', 'Cerebelum_7b_R', 'Cerebelum_8_L', 'Cerebelum_8_R', 'Cerebelum_9_L', 'Cerebelum_9_R', 'Cerebelum_10_L', 'Cerebelum_10_R']
    # 7/9 sections train: 2/9 sections test?
    df_results = run_ridge(
        embed_type=embed_type,
        dataset=dataset,
        participant=participant,
        train_sections=train_sections,
        test_sections=test_sections,
        cortex_regions=cortex_regions
    )

    print('end')


if __name__ == '__main__':
    main()