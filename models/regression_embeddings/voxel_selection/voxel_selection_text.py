import numpy as np
from scipy.linalg import block_diag

from dataset_loader.dataset import LPPDataset
from dataset_loader.section_participant_base import BaseSectionParticipant
from utils.cortical_masking import get_oxford_mask
from utils.embeddings import load_embeddings
import collections


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

    for v in range(m):
        if v % 1000 == 0:
            print('iter:', v)

        if voxel_mask[v]:
            print(v)
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



def main():
    dataset = LPPDataset()
    participant = 'sub-EN057'
    ps = BaseSectionParticipant(dataset[0])


    n_volumes = 10
    # get images
    subject_images = ps.fmri[...,:n_volumes].numpy()

    # get words per volume
    words = [ps.get_words_volume_idx(i) for i in range(n_volumes)]
    no_words = [0] + [len(l) for l in words]
    index_words = np.cumsum(no_words)


    # load embeddings
    embed_type = 'GloVe'
    df = load_embeddings(embed_type)


    # get embeddings of volumes: calculate mean if multiple words.
    # TODO: make this more neat, by actually extracting word embeeddings correctly
    encodings = np.array([df[embed_type].iloc[index_words[i]:index_words[i+1]].mean() for i in range(len(no_words)-1)])


    # Final data prep: normalize.
    X = subject_images - subject_images.mean(axis=0)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    X = np.nan_to_num(X) #fill na
    Y = encodings - encodings.mean(axis=0)
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)



    # Assuming X is your fMRI data and Y is your labels
    X_shape = X.shape
    Y_shape = Y.shape

    # Reshape the fMRI data to 2D (flattening the spatial dimensions)
    X_reshaped = X.reshape((X_shape[0] * X_shape[1] * X_shape[2], X_shape[3])).T

    # DO VOXEL SELECTION
    #load cortical mask
    # TODO: only select cortical region: no gray matter + background
    # cortex_region = 'Superior Temporal Gyrus, anterior division'
    cortex_region = 'Background'
    voxel_mask = get_oxford_mask(cortical_regions= [cortex_region])
    voxel_mask = 1- voxel_mask.get_fdata()
    voxel_mask = voxel_mask.flatten()

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
        examples = X_reshaped,
        targets= Y,
        meta = meta,
        voxel_mask  = voxel_mask
    )

    # fill nan as 0 in scores
    scores=np.nan_to_num(scores) #fill na

    # calculate mean score over all dimensions of embeddings for each voxel
    meanscores = scores.mean(axis=0)

    index_highest_voxels = find_top_n_indices(meanscores,n=5000)

    print('end')


if __name__ == '__main__':
    main()

boolmask = [True for i in range(meta['voxelsToNeighbours'].shape[0]) if np.count_nonzero(meta['voxelsToNeighbours'][i])==0  ]

boolmask = [np.count_nonzero(meta['voxelsToNeighbours'][i])==0 for i in range(meta['voxelsToNeighbours'].shape[0])]
arr = meta['voxelsToNeighbours'][np.count_nonzero(meta['voxelsToNeighbours'])==0]



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

