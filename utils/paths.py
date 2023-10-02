import os
from pathlib import Path


path = Path(r'C:\Users\tibor\Documents\thesis\code_and_data')
# when using harddrive dataset
data_path = Path(r'F:\dataset') #TODO: change to dataset, when full one downloaded

# if using local subset
# data_path = os.path.join(path,'dataset_sample') #TODO: change to dataset, when full one downloaded

annot_path =  data_path /'annotation'
fmri_path = data_path /'derivatives'
eval_path = path / 'evaluation'
deriv_path = data_path /'derivatives'

# path = r'C:\Users\tibor\Documents\thesis\code_and_data'
# # when using harddrive dataset
# data_path = r'D:\dataset' #TODO: change to dataset, when full one downloaded
#
# # if using local subset
# # data_path = os.path.join(path,'dataset_sample') #TODO: change to dataset, when full one downloaded
#
# annot_path =  os.path.join(data_path,'annotation')
# fmri_path = os.path.join(data_path,'derivatives')
# eval_path = os.path.join(path,'evaluation')
# deriv_path = os.path.join(data_path,'derivatives')