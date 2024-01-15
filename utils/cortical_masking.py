import nilearn
import numpy as np
from nilearn import datasets, image, plotting
from nilearn.image import resample_img
import nibabel as nib
from tqdm import tqdm
import torch

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


def get_destrieux_mask_depricated():
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

def get_oxford_mask(cortical_regions : list[str],select_all_regions=False):
    """Cortical regions need to be part of the atlas """

    # Load Harvard-Oxford cortical probabilistic atlas (50% threshold by default)
    harvard_oxford = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr50-1mm')

    #select all regions (except background) if specified:
    if select_all_regions:
        cortical_regions = harvard_oxford.labels
        cortical_regions.remove('Background')
    # get index of cortical regions
    cortical_indices = [harvard_oxford.labels.index(region) for region in cortical_regions]

    cortical_mask = image.math_img('img == {}'.format(cortical_indices[0]), img=harvard_oxford.maps)
    shape = cortical_mask.shape
    affine = cortical_mask.affine
    cortical_mask = cortical_mask.get_fdata()
    if cortical_regions[0] == 'Background':
        cortical_mask = 1 - cortical_mask
    for index in cortical_indices[1:]:
        if harvard_oxford.labels[index] == 'Background':
            print('background')
            cortical_mask += 1- image.math_img('img == {}'.format(index), img=harvard_oxford.maps).get_fdata()
        else:
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

def get_aal_mask(cortical_regions : list[str],select_all_regions=False):
    """Cortical regions need to be part of the atlas """

    # load aal atlas
    aal_atlas = datasets.fetch_atlas_aal()

    # AAL atlas filename
    aal_img = aal_atlas.maps
    # Load the AAL atlas image
    img_aal = image.load_img(aal_img)

    #select all regions (except background) if specified:
    if select_all_regions:
        cortical_regions = aal_atlas.labels
    # get index of cortical regions
    cortical_indices = [aal_atlas.indices[aal_atlas.labels.index(region)] for region in cortical_regions]

    # Create a mask for the prefrontal cortex from AAL atlas
    cortical_mask = image.math_img('img == {}'.format(cortical_indices[0]), img=img_aal)
    shape = cortical_mask.shape
    affine = cortical_mask.affine
    cortical_mask = cortical_mask.get_fdata()
    for index in cortical_indices[1:]:
        cortical_mask += image.math_img('img == {}'.format(index), img=img_aal).get_fdata()

    # convert to nifti imgage
    header = nib.nifti1.Nifti1Header()
    header.set_data_shape(shape)
    cortical_mask = nib.nifti1.Nifti1Image(cortical_mask, affine=affine, header=header)

    #reshape mask to mni template format
    cortical_mask = resample_mask(cortical_mask)
    return cortical_mask

def make_nifti_image_from_numpy(fmri_n):
    affine = np.array(
        [[2., 0., 0., -72.],
         [0., 2., 0., -106.],
         [0., 0., 2., -64.],
         [0., 0., 0., 1.]])
    header = nib.nifti1.Nifti1Header()
    header.set_data_shape(fmri_n.shape)
    nifti_image = nib.nifti1.Nifti1Image(fmri_n, affine=affine, header=header)
    return nifti_image
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

def mask_timeseries(timeseries:np.array,cortical_mask):
    return np.array([nilearn.masking.apply_mask(
        make_nifti_image_from_numpy(timeseries[...,i]),
        mask_img=cortical_mask)
        for i in range(timeseries.shape[3])
    ])
    # return

def mask_avg_fmri_word_dict(avg_fmri_word_dict,cortical_mask):
    """function that applies a cortical mask to the dict of average fmri volumes per word"""
    masked_avg_fmri_word_dict = {}
    for word,vol in tqdm(avg_fmri_word_dict.items()):
        masked_avg_fmri_word_dict[word] = torch.tensor(nilearn.masking.apply_mask(
            make_nifti_image_from_tensor(vol),
            mask_img=cortical_mask))

    return masked_avg_fmri_word_dict

def show_mask(mask,title = ''):
    plotting.plot_roi(mask,title=title)
    plotting.show()

if __name__ == '__main__':
    mask = get_aal_mask(['Precuneus_L', 'Precuneus_R'],select_all_regions=True)
    mask2 = get_oxford_mask(cortical_regions=['Inferior Frontal Gyrus, pars triangularis', 'Inferior Frontal Gyrus, pars opercularis'])
    mask1 = get_aal_mask(['Temporal_Sup_L', 'Temporal_Sup_R', 'Temporal_Pole_Sup_L', 'Temporal_Pole_Sup_R', 'Temporal_Mid_L', 'Temporal_Mid_R', 'Temporal_Pole_Mid_L', 'Temporal_Pole_Mid_R', 'Temporal_Inf_L', 'Temporal_Inf_R'])

    get_oxford_mask(cortical_regions=[],select_all_regions=True)
    cortical_regions = ['Frontal Pole','Occipital Pole']
    reg_mask = get_oxford_mask(cortical_regions=cortical_regions)
    bg = get_oxford_mask(cortical_regions=['Background'])
    print('done')