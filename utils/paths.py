from pathlib import Path
import os

path = Path(r'C:\Users\tibor\Documents\thesis\code_and_data')
# when using harddrive dataset
data_path = Path(r'F:\dataset')

# if using local subset
# data_path = path/'dataset_sample'
# data_path = os.path.join(path,'dataset_sample') #TODO: change to dataset, when full one downloaded

annot_path =  data_path /'annotation'
fmri_path = data_path /'derivatives'
eval_path = path / 'evaluation'
deriv_path = data_path /'derivatives'
pred_path = path / 'predictions'
