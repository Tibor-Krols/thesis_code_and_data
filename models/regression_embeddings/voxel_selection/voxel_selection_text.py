import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy.linalg import block_diag

from dataset_loader.dataset import LPPDataset
from dataset_loader.section_participant_base import BaseSectionParticipant
from utils.cortical_masking import get_aal_mask, mask_timeseries
from utils.embeddings import load_embeddings
import collections
import nibabel as nib
from utils.paths import *

def save_selected_voxel_mask(mask,
                             participant: str | list[str],
                             embed_type:str,
                             filepath:Path= data_path /'voxel_selection_masks',
                             ):
    if isinstance(participant,str):
        filepath = filepath/'per_participant'
    else:
        filepath = filepath/'across_participants'
    os.makedirs(filepath,exist_ok=True)
    filename = f"{participant}_{embed_type}_selected_voxel_mask.nii.gz"
    nib.save(mask, filepath/filename)
def load_selected_voxel_mask(participant: str | list[str],
                             embed_type:str,
                             filepath:Path= data_path /'voxel_selection_masks',
                             selection_criterium:str ='max'
                             ):
    if selection_criterium =='max':
        filepath = filepath
    elif selection_criterium == 'avg':
        filepath = filepath/'avg'

    if isinstance(participant,str):
        filepath = filepath/'per_participant'
    else:
        filepath = filepath/'across_participants'

    filename = f"{participant}_{embed_type}_selected_voxel_mask.nii.gz"

    return nib.load(filepath/filename)


def create_mask_from_voxel_selection(voxel_selection_indices, cortical_mask):
    # Get the shape of the original volume
    original_shape = (73, 90, 74)
    affine = cortical_mask.affine
    # get the indices of the original cortical mask
    indices = np.where(cortical_mask.get_fdata().flatten() > 0)[0]
    indices_selected = indices[voxel_selection_indices]

    # Create an empty mask with the original shape
    new_mask = np.zeros(original_shape)

    # Set the values at the subset indices to 1
    new_mask[np.unravel_index(indices_selected, original_shape)] = 1
    # convert to nifti imgage
    header = nib.nifti1.Nifti1Header()
    header.set_data_shape(original_shape)
    mask_selected_voxels = nib.nifti1.Nifti1Image(new_mask, affine=affine, header=header)
    return mask_selected_voxels
def find_top_n_indices(scores, n):
    # Use argsort to get the indices that would sort the array
    sorted_indices = np.argsort(scores)

    # Take the last n indices to get the top n
    top_n_indices = sorted_indices[-n:]

    return top_n_indices

def create_meta(mask_shape):
    meta = {'numberOfNeighbours': np.full(np.prod(mask_shape), 27)}

    meta['voxelsToNeighbours'] = np.zeros((np.prod(mask_shape), 27), dtype=int)

    index = 0
    for x in range(mask_shape[0]):
        for y in range(mask_shape[1]):
            for z in range(mask_shape[2]):
                neighbors = []
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            new_x, new_y, new_z = x + dx, y + dy, z + dz

                            if (
                                0 <= new_x < mask_shape[0] and
                                0 <= new_y < mask_shape[1] and
                                0 <= new_z < mask_shape[2]
                            ):
                                flattened_index = np.ravel_multi_index((new_x, new_y, new_z), mask_shape, order='F')
                                neighbors.append(flattened_index)

                # Ensure the correct number of neighbors is stored
                meta['voxelsToNeighbours'][index, :len(neighbors)] = neighbors
                index += 1

    return meta

def train_voxelwise_target_prediction_models(examples, targets, **kwargs):
    """
    FUNCTION from pereira et al. 2018 translated from matlab python by chatgpt
    Score voxels by how well a regression model trained on the voxel
    (+neighbours) predicts each column of a target matrix.

    Parameters:
    - examples: ndarray, shape (#examples, #voxels)
    - targets: ndarray, shape (#examples, #dimensionsTarget)

    Optional parameters:
    - groupby: ndarray, shape (#examples,), groups for cross-validation (defaults to 10-fold)
    - voxel_mask: ndarray, shape (#voxels,), binary mask to exclude voxels of no interest
    - meta: ndarray, use ridge regression from searchlight (voxel + 3D neighbours)
    - lambda_: float, lambda for ridge regression (default is 1)

    Returns:
    - scores: ndarray, shape (#dimensionsTarget, #voxels), per voxel correlation between target and cross-validated prediction
    """

    n, m = examples.shape
    n1, mt = targets.shape

    if n1 != n:
        print('error: targets must have as many rows as examples')
        return

    # Defaults
    meta = kwargs.get('meta', None)
    lambda_ = kwargs.get('lambda_', 1)
    voxel_mask = kwargs.get('voxel_mask', np.ones(m))
    groupby = kwargs.get('groupby', 1 + np.remainder(np.arange(1, n + 1), 10))

    # Other defaults
    use_searchlight = meta is not None

    # Precompute
    onecol = np.ones((n, 1))
    targets_z = (targets - np.mean(targets, axis=0)) / np.std(targets, axis=0)
    targets_c = targets - np.mean(targets, axis=0)
    examples_z = examples - np.mean(examples, axis=0)
    predicted = np.zeros((n, mt))
    scores = np.zeros((mt, m))

    groups = np.unique(groupby)
    n_groups = len(groups)

    for ig in range(n_groups):
        mask = groupby == groups[ig]
        indices_test = np.where(mask)[0]
        n_test = len(indices_test)
        indices_train = np.where(~mask)[0]
        n_train = len(indices_train)

        targets_per_group = targets[indices_train, :]

    for v in tqdm(range(m)):
        # if v % 1000 == 0:
        #     print('iter:', v)

        if voxel_mask[v]:
            # print(v)
            if use_searchlight:
                # nn = meta.number_of_neighbours(v)
                # nn = meta['voxelsToNeighbours'][v]
                # nn1 = nn + 1 #old way
                # nn1 = nn.shape[0]
                nn1 = meta['numberOfNeighbours'][v]
                neighbours = meta['voxelsToNeighbours'][v]
                # neighbours = [v] + list(meta.voxels_to_neighbours(v)[:nn])
                regularization_matrix = lambda_ * np.eye(nn1)
            else:
                nn = 0
                neighbours = [v]
                regularization_matrix = lambda_

            data = examples[:, neighbours]

            # Run cross-validation loop
            for ig in range(n_groups):
                tmp = data[indices_train, :]
                betas = np.linalg.solve(tmp.T @ tmp + regularization_matrix, tmp.T @ targets_c[indices_train, :])
                predicted[indices_test, :] = data[indices_test, :] @ betas

            scores[:, v] = np.sum(targets_z * (predicted - np.mean(predicted, axis=0)) / np.std(predicted, axis=0), axis=0) / (n - 1)

    return scores


def load_masked_volumes_embeddings(
        dataset:LPPDataset,
        participant:str,
        cortical_mask,
        embed_type:str,
        select_all_regions: bool = False,
        sections:list[int]=range(1,10)
):
    # initialize variables
    encodings_total = None
    subject_images_total = None
    for section in tqdm(sections):
        ps_index = dataset.get_participant_section_index(participant=participant,section=section)
        ps = BaseSectionParticipant(dataset[ps_index],embed_type=embed_type)
        # section = ps.section
        n_volumes = ps.nr_fmri_frames
        # TODO: use all volumes
        # n_volumes = 15 #small for testing purposes

        # get words per volume
        words = [ps.get_words_volume_idx(i) for i in range(n_volumes)]
        # select indices that contain words
        indices_vols_with_words = [
            i for i in range(n_volumes) if words[i] != []
        ]

        # get images
        subject_images = ps.fmri[...,indices_vols_with_words].numpy()

        # mask_images

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

    # Final data prep: normalize.
    X = subject_images_total - subject_images_total.mean(axis=0)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    # X=np.nan_to_num(X) #fill na (not needed now
    Y = encodings_total - encodings_total.mean(axis=0)
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)

    return X,Y


def main():
    embed_type = 'BERT'
    dataset = LPPDataset(embed_type=embed_type)
    participant = 'sub-EN057'
    sections = [1,2,3,4,5,6,7,9]
    cortex_regions = ['Olfactory_L']
    select_all_regions=True
    # ps = BaseSectionParticipant(dataset[0],embed_type=embed_type)
    # load mask
    cortical_mask = get_aal_mask(cortical_regions=cortex_regions,
                                 select_all_regions=select_all_regions)

    X,Y = load_masked_volumes_embeddings(
        dataset=dataset,
        participant=participant,
        cortical_mask=cortical_mask,
        embed_type=embed_type,
        select_all_regions = False,
        sections=sections
    )

    # Reshape the fMRI data to 2D (flattening the spatial dimensions)
    if len(X.shape) ==3:
        X = X.reshape((X.shape[0] * X.shape[1] * X.shape[2], X.shape[3])).T

    # DO VOXEL SELECTION
    # specify meta:
    # TODO: get the surrounding voxel selection to work
    grid_shape = (5, 5, 5)
    # Initialize meta with information about 26 neighboring voxels for each voxel
    # meta = {'number_of_neighbours': np.full(np.prod(grid_shape), 26)}
    #
    # meta['voxels_to_neighbours'] = np.zeros((np.prod(grid_shape), 26), dtype=int)
    meta = create_meta((73, 90, 74))
    # kwargs = {'meta':True,'voxel_mask':voxel_mask}
    scores = train_voxelwise_target_prediction_models(
        examples = X,
        targets= Y,
        # meta = meta, #for selecting surrounding voxels
        # voxel_mask  = voxel_mask
    )

    # fill nan as 0 in scores
    scores=np.nan_to_num(scores) #fill na

    # calculate mean score over all dimensions of embeddings for each voxel
    meanscores = scores.mean(axis=0)
    maxscores = scores.max(axis=0)

    index_highest_avg_voxels = find_top_n_indices(meanscores,n=5000)
    index_highest_voxels = find_top_n_indices(maxscores,n=5000)

    mask_selected_voxels = create_mask_from_voxel_selection(
        voxel_selection_indices = index_highest_voxels,
        cortical_mask=cortical_mask
        )
    mask_avg_selected_voxels = create_mask_from_voxel_selection(
        voxel_selection_indices=index_highest_avg_voxels,
        cortical_mask=cortical_mask
    )


    #save mask
    save_selected_voxel_mask(mask=mask_selected_voxels,
                             participant=participant,
                             embed_type=embed_type)

    save_selected_voxel_mask(mask=mask_avg_selected_voxels,
                             participant=participant,
                             embed_type=embed_type,
                             filepath=data_path /'voxel_selection_masks'/'avg'#selects average score over all embeddings per voxel
                             )
    print(index_highest_voxels)
    print('end')


if __name__ == '__main__':
    main()

# boolmask = [True for i in range(meta['voxelsToNeighbours'].shape[0]) if np.count_nonzero(meta['voxelsToNeighbours'][i])==0  ]
#
# boolmask = [np.count_nonzero(meta['voxelsToNeighbours'][i])==0 for i in range(meta['voxelsToNeighbours'].shape[0])]
# arr = meta['voxelsToNeighbours'][np.count_nonzero(meta['voxelsToNeighbours'])==0]



def adj_to_neighbor_dict(adj):
    assert hasattr(adj, "__iter__")

    neighbor_dict = collections.defaultdict(lambda: set())
    for i,j in adj:
        if i == j:
            continue
        neighbor_dict[i].add(j)
        neighbor_dict[j].add(i)
    return neighbor_dict

def get_neighbors_2d(npmatrix):
    assert len(npmatrix.shape) == 2
    I, J = range(npmatrix.shape[0]-1), range(npmatrix.shape[1]-1)
    adj_set = set(
        (npmatrix[i,j], npmatrix[i+1,j])
        for i in I
        for j in J
    ) | set(
        (npmatrix[i,j], npmatrix[i,j+1])
        for i in I
        for j in J
    )
    return adj_to_neighbor_dict(adj_set)

def get_neighbors_3d(npmatrix):
    assert len(npmatrix.shape) == 3
    I, J, K = range(npmatrix.shape[0]-1), range(npmatrix.shape[1]-1), range(npmatrix.shape[2]-1)
    adj_set = set(
        (npmatrix[i,j,k], npmatrix[i+1,j,k])
        for i in I
        for j in J
        for k in K
    ) | set(
        (npmatrix[i,j,k], npmatrix[i,j+1,k])
        for i in I
        for j in J
        for k in K
    ) | set(
        (npmatrix[i,j,k], npmatrix[i,j,k+1])
        for i in I
        for j in J
        for k in K
    )
    return adj_to_neighbor_dict(adj_set)

