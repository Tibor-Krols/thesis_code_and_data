import random

from dataset_loader.dataset import LPPDataset
from utils.paths import *
from sklearn.model_selection import train_test_split
def train_test_validation_split_percentage(dataset, train_perc:float=.8,test_perc:float = .1,val_perc:float = .1)->tuple[list[int],list[int],list[int]]:
    """
    Function that takes train test validation percentages
    since the split is done based on the nr of sections and nr of participants.
    because of rounding, the total perecentage of samples differs from specified percentages
    :param dataset:
    :param train_perc:
    :param test_perc:
    :param val_perc:
    :return:
    """
    participants = dataset.participants
    section_nrs = dataset.section_nrs
    # split participants
    train_participants, test_and_validation_participants = train_test_split(participants, test_size=test_perc + val_perc,
                                                            random_state=42)
    test_participants, validation_participants = train_test_split(test_and_validation_participants,
                                                  test_size=val_perc / (test_perc + val_perc),
                                                  random_state=42)
    # split sections
    train_sections, test_and_validation_sections = train_test_split(section_nrs, test_size=test_perc + val_perc,
                                                            random_state=42)
    test_sections, validation_sections = train_test_split(test_and_validation_sections,
                                                  test_size=val_perc / (test_perc + val_perc),
                                                  random_state=42)
    train_indices = [i for i,s in enumerate(dataset.samples) if s[1] in train_participants and s[3] in train_sections]
    test_indices = [i for i,s in enumerate(dataset.samples) if s[1] in test_participants or s[3] in test_sections]
    validation_indices = [i for i,s in enumerate(dataset.samples) if s[1] in validation_participants or s[3] in validation_sections]
    return train_indices,test_indices,validation_indices

# def train_test_validation_split(dataset, train_perc:float=.8,test_perc:float = .1,val_perc:float = .1)->tuple[list[int],list[int],list[int]]:

def train_test_validation_split(dataset,Ntrain_participants=46,Ntrain_sections=7,Ntest_participants =1,Ntest_sections = 1,Nvalidation_participants =1,Nvalidation_sections=1)->tuple[list[int],list[int],list[int]]:
    """
    split based on specified nr of participants and sections
    default train,test validation:
        for participants 46,1,1
        for sections, 7,1,1

    with this default, train, test validation is approximately 74,13,13 %
    :param dataset:
    :param Ntrain_participants:
    :param Ntrain_sections:
    :param Ntest_participants:
    :param Ntest_sections:
    :param Nvalidation_participants:
    :param Nvalidation_sections:
    :return:
    """
    random.seed(42)
    participants = dataset.participants
    section_nrs = dataset.section_nrs
    nr_of_sections = dataset.nr_of_sections
    nr_participants = dataset.nr_participants
    samples = dataset.samples
    type = ['train','test','validation']
    nr_participants_list = [Ntrain_participants,Ntest_participants,Nvalidation_participants]
    nr_sections_list = [Ntrain_sections,Ntest_sections,Nvalidation_sections]
    dict_percentages_participants = {}
    dict_percentages_sections = {}
    for fold,n_part,n_sect in zip(type,nr_participants_list,nr_sections_list):
        perc_participants = n_part/nr_participants
        perc_sections = n_sect/nr_of_sections
        dict_percentages_participants[fold]=perc_participants
        dict_percentages_sections[fold] = perc_sections

    # split participants
    train_participants, test_and_validation_participants = train_test_split(participants, test_size=dict_percentages_participants['test'] + dict_percentages_participants['validation'],
                                                            random_state=42)
    test_participants, validation_participants = train_test_split(test_and_validation_participants,
                                                  test_size=dict_percentages_participants['validation'] / (dict_percentages_participants['test'] + dict_percentages_participants['validation']),
                                                  random_state=42)
    # split sections
    train_sections, test_and_validation_sections = train_test_split(section_nrs, test_size=dict_percentages_sections['test'] + dict_percentages_sections['validation'],
                                                            random_state=42)
    test_sections, validation_sections = train_test_split(test_and_validation_sections,
                                                  test_size=dict_percentages_sections['validation'] / (dict_percentages_sections['test'] + dict_percentages_sections['validation']),
                                                  random_state=42)

    train_indices = [i for i, s in enumerate(samples) if s[1] in train_participants and s[3] in train_sections]
    test_indices = [i for i, s in enumerate(samples) if
                    (s[1] in test_participants or s[3] in test_sections) and s[1] not in validation_participants
                     ]
    validation_indices = [i for i, s in enumerate(samples) if
                          (s[1] in validation_participants or s[3] in validation_sections) and s[1] not in test_participants
                           ]

    return train_indices,test_indices,validation_indices

def get_train_test_sections(section_nrs = [i for i in range(1,10)], Ntrain_sections=8,Ntest_sections =1):
    random.seed(42)
    nr_of_sections = len(section_nrs)
    type = ['train','test']
    nr_sections_list = [Ntrain_sections,Ntest_sections]
    dict_percentages_sections = {}
    for fold,n_sect in zip(type,nr_sections_list):
        perc_sections = n_sect/nr_of_sections
        dict_percentages_sections[fold] = perc_sections
    train_sections, test_sections = train_test_split(section_nrs, test_size=dict_percentages_sections['test'],
                                                            random_state=42)
    return train_sections,test_sections
def train_test_split_lpp(dataset,Ntrain_participants=44,Ntrain_sections=8,Ntest_participants =4,Ntest_sections = 1)->tuple[list[int],list[int]]:
    """
    split based on specified nr of participants and sections
    default train,test validation:
        for participants 46,1,1
        for sections, 7,1,1

    with this default, train, test validation is approximately 74,13,13 %
    :param dataset:
    :param Ntrain_participants:
    :param Ntrain_sections:
    :param Ntest_participants:
    :param Ntest_sections:
    :param Nvalidation_participants:
    :param Nvalidation_sections:
    :return:
    """
    random.seed(42)
    participants = dataset.participants
    section_nrs = dataset.section_nrs
    nr_of_sections = dataset.nr_of_sections
    nr_participants = dataset.nr_participants
    samples = dataset.samples
    type = ['train','test']
    nr_participants_list = [Ntrain_participants,Ntest_participants]
    nr_sections_list = [Ntrain_sections,Ntest_sections]
    dict_percentages_participants = {}
    dict_percentages_sections = {}
    for fold,n_part,n_sect in zip(type,nr_participants_list,nr_sections_list):
        perc_participants = n_part/nr_participants
        perc_sections = n_sect/nr_of_sections
        dict_percentages_participants[fold]=perc_participants
        dict_percentages_sections[fold] = perc_sections

    # split participants
    train_participants, test_participants = train_test_split(participants, test_size=dict_percentages_participants['test'],
                                                            random_state=42)
    # test_participants, validation_participants = train_test_split(test_and_validation_participants,
    #                                               test_size=dict_percentages_participants['validation'] / (dict_percentages_participants['test'] + dict_percentages_participants['validation']),
    #                                               random_state=42)
    # split sections
    train_sections, test_sections = train_test_split(section_nrs, test_size=dict_percentages_sections['test'],
                                                            random_state=42)
    # test_sections, validation_sections = train_test_split(test_and_validation_sections,
    #                                               test_size=dict_percentages_sections['validation'] / (dict_percentages_sections['test'] + dict_percentages_sections['validation']),
    #                                               random_state=42)

    train_indices = [i for i, s in enumerate(samples) if s[1] in train_participants and s[3] in train_sections]
    test_indices = [i for i, s in enumerate(samples) if
                    (s[1] in test_participants or s[3] in test_sections)
                     ]
    # validation_indices = [i for i, s in enumerate(samples) if
    #                       (s[1] in validation_participants or s[3] in validation_sections) and s[1] not in test_participants
    #                        ]
    return train_indices,test_indices


def main():
    dataset = LPPDataset(deriv_path)
    # use if need train test validation split:
    # train_indices,test_indices,validation_indices = train_test_validation_split(dataset=dataset)
    # train_indices1,test_indices1,validation_indices1 = train_test_validation_split_percentage(dataset=dataset)
    # use if need train test split (can be used iteratively to create validation fold)
    train_indices,test_indices = train_test_split_lpp(dataset)
    print('end')
if __name__ == '__main__':
    main()
