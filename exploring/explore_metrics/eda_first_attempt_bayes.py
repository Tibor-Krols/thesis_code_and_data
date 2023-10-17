import pandas as pd

from utils.paths import *


bayes_path = eval_path / 'metrics' / 'bayesian'/'metrics_bayes_vol_pred_sub-EN057_mse.csv'
base_path = eval_path / 'metrics_backup'/'baseline_metrics_full_book_bertScore_randomstateFixed.csv'
bayes_path_corr = eval_path / 'metrics' / 'bayesian'/'metrics_bayes_vol_pred_sub-EN057.csv'

base = pd.read_csv(base_path)
bayes = pd.read_csv(bayes_path)
bayes_corr = pd.read_csv(bayes_path_corr)
sum_base = base.describe()
sum_bayes = bayes.describe()


print('end')