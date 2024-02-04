from dataset_loader.dataset import LPPDataset
from dataset_loader.section_participant_base import BaseSectionParticipant
from evaluation.calculate_metrics import save_bayesian_volume_metrics_participant
from models.bayesian.bayesian_prediction import get_prior_dict, run_predictions_participant_section, \
    get_prior_dict_trainset
from models.bayesian.fmri_averages_per_participant import load_averages_participant, save_averages_participant
from utils import file_saving
from utils.cortical_masking import  make_nifti_image_from_tensor, mask_avg_fmri_word_dict, \
    load_cortical_atlas_flexible
from utils.paths import *
from tqdm import tqdm

# vol3 = avg_fmri_word_dict['drink']
# masked_vol3 = nilearn.masking.apply_mask(make_nifti_image_from_tensor(vol3), mask_img=cortical_mask)
# np.linalg.norm(masked_vol3-masked_vol)



def main():
    participant = 'sub-EN058'
    cortical_regions = ['Angular Gyrus',
            'Cerebelum_Crus1_L', 'Cerebelum_Crus1_R', 'Cerebelum_Crus2_L', 'Cerebelum_Crus2_R', 'Cerebelum_3_L',
            'Cerebelum_3_R', 'Cerebelum_4_5_L', 'Cerebelum_4_5_R', 'Cerebelum_6_L', 'Cerebelum_6_R', 'Cerebelum_7b_L',
            'Cerebelum_7b_R', 'Cerebelum_8_L', 'Cerebelum_8_R', 'Cerebelum_9_L', 'Cerebelum_9_R', 'Cerebelum_10_L',
            'Cerebelum_10_R',
            'Superior Temporal Gyrus, anterior division', 'Superior Temporal Gyrus, posterior division',
            'Temporal_Sup_L', 'Temporal_Pole_Sup_L', 'Temporal_Mid_L', 'Temporal_Pole_Mid_L', 'Temporal_Inf_L',
            'Temporal_Sup_R', 'Temporal_Pole_Sup_R', 'Temporal_Mid_R', 'Temporal_Pole_Mid_R', 'Temporal_Inf_R'
        ]

    dataset = LPPDataset()
    participant_indices = dataset.get_participant_samples_indices(participant)
    # train_indices = participant_indices[:8]
    train_indices = participant_indices[:7] + [participant_indices[-1]]
    # test_indices = [participant_indices[-1]]
    test_indices = [participant_indices[7]] #select section 8
    # test_indices = [ participant_indices[0]] #use already 'seen' section to test if overfitting
    ps_idx = test_indices[0]
    ps = BaseSectionParticipant(dataset[ps_idx], include_volume_words_dict=True)

    # similarity_type = 'mse'
    similarity_type = 'mse'
    # prior
    # prior_dict = get_prior_dict()
    prior_dict = get_prior_dict_trainset()


    # specify cortical areas and construct cortical mask
    cortical_mask = load_cortical_atlas_flexible(cortical_regions)
    #mask avg fmri images if mas provided:
    # load masked images #TODO: automatically load if present, compute if not
    try:
        masked_avg_fmri_word_dict = load_averages_participant(participant, cortical_regions=cortical_regions)
    except:
        # compute masked images (if needed)
        avg_fmri_word_dict = load_averages_participant(participant)
        masked_avg_fmri_word_dict = mask_avg_fmri_word_dict(avg_fmri_word_dict,cortical_mask)
        save_averages_participant(avg_dict=masked_avg_fmri_word_dict,participant=participant,cortical_regions=cortical_regions)

    # run predictions bayesian fmri
    df_pred = run_predictions_participant_section(
        ps=ps,
        prior_dict=prior_dict,
        avg_fmri_word_dict=masked_avg_fmri_word_dict,
        similarity_type=similarity_type,
        cortical_mask=cortical_mask

    )
    filename = f"bayes_vol_pred_{ps.participant}_{similarity_type}"
    if cortical_regions is not None:
        region_names = '_'.join(cortical_regions)
        if len(region_names) > 150:
            region_names = cortical_regions[0] + '_and_more'
        filename = f"bayes_vol_pred_{ps.participant}_section{ps.section}_{similarity_type}_masked_{region_names}"

    else:
        filename = f"bayes_vol_pred_{ps.participant}_section{ps.section}_{similarity_type}"
    pred_file_path = pred_path / 'bayesian'/'per_participant'
    # save predictions
    file_saving.save_df(
        df_pred,
        filename=filename,
        save_path=pred_file_path
    )

    # calculate_metrics
    df_metrics = save_bayesian_volume_metrics_participant(
        filename=filename,
        filepath=pred_file_path,
    )
    print('done')

if __name__ == '__main__':
    main()