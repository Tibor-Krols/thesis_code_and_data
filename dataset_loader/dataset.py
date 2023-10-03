import os
import torch
from torch.utils.data import Dataset
from utils.paths import *
import nibabel as nib
from utils.load_participant_section import load_all_sections_timestamps,map_words_to_volumes,load_section_timestamps


class LPPDataset(Dataset):
    def __init__(self, root_dir = deriv_path, lang = 'EN',use_zip = True):
        """
        Initialize the hierarchical dataset for LPP

        Args:
            root_dir (str): Root directory of the dataset.
        """
        super().__init__()
        self.root_dir = root_dir
        self.participant_folders = sorted(os.listdir(root_dir))
        self.participant_folders = [f for f in self.participant_folders if f.__contains__(lang)]
        # Create a list of (participant_folder, section_folder) pairs
        self.samples = []
        for participant_folder in self.participant_folders:
            participant_path = os.path.join(root_dir, participant_folder,'func')

            files = sorted(os.listdir(participant_path))
            if use_zip:
                files = [f for f in files if f.endswith('.gz')]  # if using zipped files
            else:
                files = [f for f in files if f.endswith('.nii')]  # if using unzipped files

            for section_nr,file in enumerate(files):
                # file_path = os.path.join(participant_path,file)
                self.samples.append((root_dir, participant_folder, file,section_nr+1))
        self.nr_participants = len(self.get_participants())
        self.participants = self.get_participants()
        self.section_nrs = list(set([s[3] for s in self.samples]))
        self.nr_of_sections = len(self.section_nrs)
        self.nr_samples = len(self.samples)
    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, index):
        """
        Return a specific sample given an index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            sample (Tensor): The data sample as a PyTorch tensor.
            participant (str): Name of the participant.
            section (str): Name of the section/experiment.
        """
        data_path, participant, section_file,section_nr = self.samples[index]
        participant_section_files = sorted([f[2] for f in self.samples if f[1].__contains__(participant) ])
        section_nr = participant_section_files.index(section_file) + 1
        file_path = os.path.join(data_path,participant,'func',section_file)
        data = nib.load(file_path).get_fdata()

        # You can convert data to a PyTorch tensor if needed
        data = torch.Tensor(data)

        sections_timestamps_dict = load_section_timestamps(section = section_nr,as_timedelta=False)
        sections_timestamps_dict = map_words_to_volumes(sections_timestamps_dict)
        labels = sections_timestamps_dict[section_nr]
        # return data, participant,section
        return {'cog_sequence': data, 'subject': participant, 'labels': labels, 'section':section_nr}

    def get_participants(self):
        participants =  [s[1] for s in self.samples]
        return list(set(participants))

    def get_participant_section_data(self,participant:str,section:int):
        participant_files = sorted([f[2] for f in self.samples if f[1].__contains__(participant) ])
        section_file =  participant_files[section-1]
        # section_nrv2 =
        # section_nr = participant_files.index(section_file) + 1
        index = [i for i in range(len(self.samples)) if self.samples[i][1] == participant and self.samples[i][2]==section_file]
        return self[index[0]] # use getitem function to return the data

def main():
    dataset = LPPDataset(deriv_path)
    dataset.get_participant_section_data('sub-EN083', 1)

    participants = dataset.get_participants()
    # Access data samples, participant, and section using indexing
    # sample, participant, section = dataset[0] old tuple format
    section_participant = dataset[0]
    print('end')

if __name__ == '__main__':
    main()


