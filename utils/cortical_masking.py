import nilearn
from dataset_loader.dataset import LPPDataset
from dataset_loader.section_participant_base import BaseSectionParticipant
from nilearn.input_data import NiftiMasker
import numpy as np
from nilearn import datasets, image, plotting
from nilearn.image import resample_img
import nibabel as nib

from models.bayesian.fmri_averages_per_participant import load_averages_participant




def resample_mask(mask):
    desired_affine = np.array(
        [[2., 0., 0., -72.],
         [0., 2., 0., -106.],
         [0., 0., 2., -64.],
         [0., 0., 0., 1.]]
    )
    desired_shape = (73, 90, 74)
    # desired_shape = vol.shape
    # desired_affine = vol.affine

    # Resample the mask to the desired shape and affine
    resampled_mask = resample_img(mask, target_shape=desired_shape, target_affine=desired_affine,interpolation='nearest')

    # Display the resampled mask
    plotting.plot_roi(resampled_mask, title=f' Mask Resampled to {desired_shape}')
    plotting.show()
    return resampled_mask


def get_destrieux_mask():
    """"needs refinement if used"""

    # Load Destrieux atlas
    destrieux = datasets.fetch_atlas_destrieux_2009()

    # Assuming the prefrontal cortex is defined in the Destrieux atlas as index 14
    prefrontal_index_destrieux = 14

    # Extract the mask for the prefrontal cortex from the Destrieux atlas
    prefrontal_mask_destrieux = image.math_img('img == {}'.format(prefrontal_index_destrieux), img=destrieux['maps'])

    # Plot the prefrontal mask from Destrieux Atlas
    plotting.plot_roi(prefrontal_mask_destrieux, title='Prefrontal Cortex Mask (Destrieux)')
    plotting.show()
    return prefrontal_mask_destrieux

def get_oxford_mask(cortical_regions : list[str]):
    """Cortical regions need to be part of the atlas """

    # Load Harvard-Oxford cortical probabilistic atlas (50% threshold by default)
    harvard_oxford = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr50-1mm')

    # get index of cortical regions
    cortical_indices = [harvard_oxford.labels.index(region) for region in cortical_regions]

    cortical_mask = image.math_img('img == {}'.format(cortical_indices[0]), img=harvard_oxford.maps)
    shape = cortical_mask.shape
    affine = cortical_mask.affine
    cortical_mask = cortical_mask.get_fdata()
    for index in cortical_indices[1:]:
        cortical_mask += image.math_img('img == {}'.format(index), img=harvard_oxford.maps).get_fdata()

    # convert to nifti imgage
    header = nib.nifti1.Nifti1Header()
    header.set_data_shape(shape)
    cortical_mask = nib.nifti1.Nifti1Image(cortical_mask, affine=affine, header=header)


    cortical_mask = resample_mask(cortical_mask)
    # Plot the cortical mask from Harvard-Oxford Atlas
    # plotting.plot_roi(cortical_mask, title=str(cortical_regions))
    # plotting.show()
    return cortical_mask
def get_aal_mask():
    """needs refinement if used"""
    # Load AAL atlas provided by Nilearn
    aal_atlas = datasets.fetch_atlas_aal()

    # AAL atlas filename
    aal_img = aal_atlas.maps

    # Load the AAL atlas image
    img_aal = image.load_img(aal_img)

    # View the labels available in the AAL atlas
    # print(aal_atlas.labels)

    # Extract the label indices corresponding to the prefrontal cortex
    # prefrontal_indices = [2, 3, 4, 5, 6]  # Update with the appropriate indices for the prefrontal cortex
    prefrontal_indices = list(range(16))

    # Create a mask for the prefrontal cortex from AAL atlas
    prefrontal_mask = image.math_img('img == {}'.format(prefrontal_indices[0]), img=img_aal)
    for index in prefrontal_indices[1:]:
        prefrontal_mask = image.math_img('np.logical_or(img == {}, mask)'.format(index), img=img_aal,
                                         mask=prefrontal_mask)

    # Plot the prefrontal mask
    plotting.plot_roi(prefrontal_mask, title='Prefrontal Cortex Mask (AAL)')
    plotting.show()
    return prefrontal_mask

def make_nifti_image_from_tensor(fmri_t):
    affine = np.array(
        [[2., 0., 0., -72.],
         [0., 2., 0., -106.],
         [0., 0., 2., -64.],
         [0., 0., 0., 1.]])
    header = nib.nifti1.Nifti1Header()
    header.set_data_shape(fmri_t.shape)
    nifti_image = nib.nifti1.Nifti1Image(fmri_t.numpy(), affine=affine, header=header)
    return nifti_image


def mask_avg_fmri_word_dict(avg_fmri_word_dict,cortical_mask):
    masked_avg_fmri_word_dict = {}
    for word,vol in tqdm(avg_fmri_word_dict.items()):
        masked_avg_fmri_word_dict[word] = nilearn.masking.apply_mask(
            make_nifti_image_from_tensor(vol),
            mask_img=cortical_mask)
    return masked_avg_fmri_word_dict