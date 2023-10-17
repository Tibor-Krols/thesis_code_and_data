from dataset_loader.dataset import LPPDataset
from dataset_loader.section_participant_base import BaseSectionParticipant
from evaluation.calculate_metrics import save_bayesian_volume_metrics_participant
from models.bayesian.bayesian_prediction import get_prior_dict, run_predictions_participant_section
from models.bayesian.fmri_averages_per_participant import load_averages_participant
from utils import file_saving
from utils.paths import *


def main():
    participant = 'sub-EN057'

    dataset = LPPDataset()
    participant_indices = dataset.get_participant_samples_indices(participant)
    train_indices = participant_indices[:8]
    test_indices = [participant_indices[-1]]
    ps_idx = test_indices[0]
    ps = BaseSectionParticipant(dataset[ps_idx], include_volume_words_dict=True)
    avg_fmri_word_dict = load_averages_participant(participant)

    similarity_type = 'mse'
    # prior
    prior_dict = get_prior_dict()

    # run predictions bayesian fmri
    df_pred = run_predictions_participant_section(ps, prior_dict, avg_fmri_word_dict, similarity_type)
    filename = f"bayes_vol_pred_{ps.participant}_{similarity_type}"
    pred_file_path = pred_path / 'bayesian'
    # save predictions
    file_saving.save_df(
        df_pred,
        filename=filename,
        save_path=pred_file_path
    )

    # calculate_metrics
    save_bayesian_volume_metrics_participant(
        filename=filename,
        filepath=pred_file_path,
    )

if __name__ == '__main__':
    main()