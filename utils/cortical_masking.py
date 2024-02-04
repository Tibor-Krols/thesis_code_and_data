import nilearn
import numpy as np
from nilearn import datasets, image, plotting
from nilearn.image import resample_img
import nibabel as nib
from tqdm import tqdm
import torch


def load_cortical_atlas_flexible(cortical_regions):
    # load from relevant atlas
    atlas_type = check_atlas_type(cortical_regions)
    if atlas_type == 'both':
        cortical_mask = load_mask_both_atlasses(cortical_regions=cortical_regions)
    elif atlas_type == 'harvard_oxford':
        cortical_mask = get_oxford_mask(cortical_regions=cortical_regions)
    elif atlas_type == 'aal':
        cortical_mask = get_aal_mask(cortical_regions=cortical_regions)

    return cortical_mask
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
def load_mask_both_atlasses(cortical_regions):
    # split cortical labels
    # currently hardcoded to speed up runs and not load labels every time
    labels_harvard_oxford = ['Background', 'Frontal Pole', 'Insular Cortex', 'Superior Frontal Gyrus', 'Middle Frontal Gyrus', 'Inferior Frontal Gyrus, pars triangularis', 'Inferior Frontal Gyrus, pars opercularis', 'Precentral Gyrus', 'Temporal Pole', 'Superior Temporal Gyrus, anterior division', 'Superior Temporal Gyrus, posterior division', 'Middle Temporal Gyrus, anterior division', 'Middle Temporal Gyrus, posterior division', 'Middle Temporal Gyrus, temporooccipital part', 'Inferior Temporal Gyrus, anterior division', 'Inferior Temporal Gyrus, posterior division', 'Inferior Temporal Gyrus, temporooccipital part', 'Postcentral Gyrus', 'Superior Parietal Lobule', 'Supramarginal Gyrus, anterior division', 'Supramarginal Gyrus, posterior division', 'Angular Gyrus', 'Lateral Occipital Cortex, superior division', 'Lateral Occipital Cortex, inferior division', 'Intracalcarine Cortex', 'Frontal Medial Cortex', 'Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)', 'Subcallosal Cortex', 'Paracingulate Gyrus', 'Cingulate Gyrus, anterior division', 'Cingulate Gyrus, posterior division', 'Precuneous Cortex', 'Cuneal Cortex', 'Frontal Orbital Cortex', 'Parahippocampal Gyrus, anterior division', 'Parahippocampal Gyrus, posterior division', 'Lingual Gyrus', 'Temporal Fusiform Cortex, anterior division', 'Temporal Fusiform Cortex, posterior division', 'Temporal Occipital Fusiform Cortex', 'Occipital Fusiform Gyrus', 'Frontal Operculum Cortex', 'Central Opercular Cortex', 'Parietal Operculum Cortex', 'Planum Polare', "Heschl's Gyrus (includes H1 and H2)", 'Planum Temporale', 'Supracalcarine Cortex', 'Occipital Pole']
    labels_aal = ['Precentral_L', 'Precentral_R', 'Frontal_Sup_L', 'Frontal_Sup_R', 'Frontal_Sup_Orb_L', 'Frontal_Sup_Orb_R', 'Frontal_Mid_L', 'Frontal_Mid_R', 'Frontal_Mid_Orb_L', 'Frontal_Mid_Orb_R', 'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R', 'Frontal_Inf_Tri_L', 'Frontal_Inf_Tri_R', 'Frontal_Inf_Orb_L', 'Frontal_Inf_Orb_R', 'Rolandic_Oper_L', 'Rolandic_Oper_R', 'Supp_Motor_Area_L', 'Supp_Motor_Area_R', 'Olfactory_L', 'Olfactory_R', 'Frontal_Sup_Medial_L', 'Frontal_Sup_Medial_R', 'Frontal_Med_Orb_L', 'Frontal_Med_Orb_R', 'Rectus_L', 'Rectus_R', 'Insula_L', 'Insula_R', 'Cingulum_Ant_L', 'Cingulum_Ant_R', 'Cingulum_Mid_L', 'Cingulum_Mid_R', 'Cingulum_Post_L', 'Cingulum_Post_R', 'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L', 'ParaHippocampal_R', 'Amygdala_L', 'Amygdala_R', 'Calcarine_L', 'Calcarine_R', 'Cuneus_L', 'Cuneus_R', 'Lingual_L', 'Lingual_R', 'Occipital_Sup_L', 'Occipital_Sup_R', 'Occipital_Mid_L', 'Occipital_Mid_R', 'Occipital_Inf_L', 'Occipital_Inf_R', 'Fusiform_L', 'Fusiform_R', 'Postcentral_L', 'Postcentral_R', 'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L', 'Parietal_Inf_R', 'SupraMarginal_L', 'SupraMarginal_R', 'Angular_L', 'Angular_R', 'Precuneus_L', 'Precuneus_R', 'Paracentral_Lobule_L', 'Paracentral_Lobule_R', 'Caudate_L', 'Caudate_R', 'Putamen_L', 'Putamen_R', 'Pallidum_L', 'Pallidum_R', 'Thalamus_L', 'Thalamus_R', 'Heschl_L', 'Heschl_R', 'Temporal_Sup_L', 'Temporal_Sup_R', 'Temporal_Pole_Sup_L', 'Temporal_Pole_Sup_R', 'Temporal_Mid_L', 'Temporal_Mid_R', 'Temporal_Pole_Mid_L', 'Temporal_Pole_Mid_R', 'Temporal_Inf_L', 'Temporal_Inf_R', 'Cerebelum_Crus1_L', 'Cerebelum_Crus1_R', 'Cerebelum_Crus2_L', 'Cerebelum_Crus2_R', 'Cerebelum_3_L', 'Cerebelum_3_R', 'Cerebelum_4_5_L', 'Cerebelum_4_5_R', 'Cerebelum_6_L', 'Cerebelum_6_R', 'Cerebelum_7b_L', 'Cerebelum_7b_R', 'Cerebelum_8_L', 'Cerebelum_8_R', 'Cerebelum_9_L', 'Cerebelum_9_R', 'Cerebelum_10_L', 'Cerebelum_10_R', 'Vermis_1_2', 'Vermis_3', 'Vermis_4_5', 'Vermis_6', 'Vermis_7', 'Vermis_8', 'Vermis_9', 'Vermis_10']
    harvard_oxford = list(set(cortical_regions) & set(labels_harvard_oxford))
    aal = list(set(cortical_regions) & set(labels_aal))

    # load both masks
    cortical_mask_harvard_oxford = get_oxford_mask(cortical_regions=harvard_oxford)
    cortical_mask_aal = get_aal_mask(cortical_regions=aal)

    # combine masks
    combined_mask = np.logical_or(cortical_mask_harvard_oxford.get_fdata(), cortical_mask_aal.get_fdata())
    # combined = cortical_mask_harvard_oxford.get_fdata() + cortical_mask_aal.get_fdata()
    header = nib.nifti1.Nifti1Header()
    header.set_data_shape(cortical_mask_aal.shape)
    cortical_mask = nib.nifti1.Nifti1Image(
        combined_mask,
        affine=cortical_mask_aal.affine,
        header=header)
    show_mask(cortical_mask,'combined')
    return cortical_mask

if __name__ == '__main__':
    mask = get_aal_mask(['Precuneus_L', 'Precuneus_R'],select_all_regions=True)
    mask2 = get_oxford_mask(cortical_regions=['Inferior Frontal Gyrus, pars triangularis', 'Inferior Frontal Gyrus, pars opercularis'])
    mask1 = get_aal_mask(['Temporal_Sup_L', 'Temporal_Sup_R', 'Temporal_Pole_Sup_L', 'Temporal_Pole_Sup_R', 'Temporal_Mid_L', 'Temporal_Mid_R', 'Temporal_Pole_Mid_L', 'Temporal_Pole_Mid_R', 'Temporal_Inf_L', 'Temporal_Inf_R'])

    get_oxford_mask(cortical_regions=[],select_all_regions=True)
    cortical_regions = ['Frontal Pole','Occipital Pole']
    reg_mask = get_oxford_mask(cortical_regions=cortical_regions)
    bg = get_oxford_mask(cortical_regions=['Background'])
    print('done')

    #test different masks:
    temporal_masks = ['Temporal_Sup_L', 'Temporal_Sup_R', 'Temporal_Pole_Sup_L', 'Temporal_Pole_Sup_R', 'Temporal_Mid_L', 'Temporal_Mid_R', 'Temporal_Pole_Mid_L', 'Temporal_Pole_Mid_R', 'Temporal_Inf_L', 'Temporal_Inf_R']
    temporal_L = [m for m in temporal_masks if '_L' in m]
    temporal_R = [m for m in temporal_masks if '_R' in m]

    mask = get_aal_mask(temporal_R)
    show_mask(mask, 'all temporal')
    for region in temporal_masks:
        mask = get_aal_mask([region])
        show_mask(mask,region)


    temporal_ho = ['Temporal Pole', 'Superior Temporal Gyrus, anterior division', 'Superior Temporal Gyrus, posterior division', 'Middle Temporal Gyrus, anterior division', 'Middle Temporal Gyrus, posterior division', 'Middle Temporal Gyrus, temporooccipital part', 'Inferior Temporal Gyrus, anterior division', 'Inferior Temporal Gyrus, posterior division', 'Inferior Temporal Gyrus, temporooccipital part', 'Temporal Fusiform Cortex, anterior division', 'Temporal Fusiform Cortex, posterior division', 'Temporal Occipital Fusiform Cortex', 'Planum Temporale']
    mask = get_oxford_mask(temporal_ho)
    show_mask(mask, 'all temporal')

    for region in temporal_ho:
        mask = get_oxford_mask([region])
        show_mask(mask,region)


def check_atlas_type(cortical_regions)->str:
    # TODO: remove hardcoding and load labels
    # currently hardcoded to speed up runs and not load labels every time
    labels_harvard_oxford = ['Background', 'Frontal Pole', 'Insular Cortex', 'Superior Frontal Gyrus', 'Middle Frontal Gyrus', 'Inferior Frontal Gyrus, pars triangularis', 'Inferior Frontal Gyrus, pars opercularis', 'Precentral Gyrus', 'Temporal Pole', 'Superior Temporal Gyrus, anterior division', 'Superior Temporal Gyrus, posterior division', 'Middle Temporal Gyrus, anterior division', 'Middle Temporal Gyrus, posterior division', 'Middle Temporal Gyrus, temporooccipital part', 'Inferior Temporal Gyrus, anterior division', 'Inferior Temporal Gyrus, posterior division', 'Inferior Temporal Gyrus, temporooccipital part', 'Postcentral Gyrus', 'Superior Parietal Lobule', 'Supramarginal Gyrus, anterior division', 'Supramarginal Gyrus, posterior division', 'Angular Gyrus', 'Lateral Occipital Cortex, superior division', 'Lateral Occipital Cortex, inferior division', 'Intracalcarine Cortex', 'Frontal Medial Cortex', 'Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)', 'Subcallosal Cortex', 'Paracingulate Gyrus', 'Cingulate Gyrus, anterior division', 'Cingulate Gyrus, posterior division', 'Precuneous Cortex', 'Cuneal Cortex', 'Frontal Orbital Cortex', 'Parahippocampal Gyrus, anterior division', 'Parahippocampal Gyrus, posterior division', 'Lingual Gyrus', 'Temporal Fusiform Cortex, anterior division', 'Temporal Fusiform Cortex, posterior division', 'Temporal Occipital Fusiform Cortex', 'Occipital Fusiform Gyrus', 'Frontal Operculum Cortex', 'Central Opercular Cortex', 'Parietal Operculum Cortex', 'Planum Polare', "Heschl's Gyrus (includes H1 and H2)", 'Planum Temporale', 'Supracalcarine Cortex', 'Occipital Pole']
    labels_aal = ['Precentral_L', 'Precentral_R', 'Frontal_Sup_L', 'Frontal_Sup_R', 'Frontal_Sup_Orb_L', 'Frontal_Sup_Orb_R', 'Frontal_Mid_L', 'Frontal_Mid_R', 'Frontal_Mid_Orb_L', 'Frontal_Mid_Orb_R', 'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R', 'Frontal_Inf_Tri_L', 'Frontal_Inf_Tri_R', 'Frontal_Inf_Orb_L', 'Frontal_Inf_Orb_R', 'Rolandic_Oper_L', 'Rolandic_Oper_R', 'Supp_Motor_Area_L', 'Supp_Motor_Area_R', 'Olfactory_L', 'Olfactory_R', 'Frontal_Sup_Medial_L', 'Frontal_Sup_Medial_R', 'Frontal_Med_Orb_L', 'Frontal_Med_Orb_R', 'Rectus_L', 'Rectus_R', 'Insula_L', 'Insula_R', 'Cingulum_Ant_L', 'Cingulum_Ant_R', 'Cingulum_Mid_L', 'Cingulum_Mid_R', 'Cingulum_Post_L', 'Cingulum_Post_R', 'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L', 'ParaHippocampal_R', 'Amygdala_L', 'Amygdala_R', 'Calcarine_L', 'Calcarine_R', 'Cuneus_L', 'Cuneus_R', 'Lingual_L', 'Lingual_R', 'Occipital_Sup_L', 'Occipital_Sup_R', 'Occipital_Mid_L', 'Occipital_Mid_R', 'Occipital_Inf_L', 'Occipital_Inf_R', 'Fusiform_L', 'Fusiform_R', 'Postcentral_L', 'Postcentral_R', 'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L', 'Parietal_Inf_R', 'SupraMarginal_L', 'SupraMarginal_R', 'Angular_L', 'Angular_R', 'Precuneus_L', 'Precuneus_R', 'Paracentral_Lobule_L', 'Paracentral_Lobule_R', 'Caudate_L', 'Caudate_R', 'Putamen_L', 'Putamen_R', 'Pallidum_L', 'Pallidum_R', 'Thalamus_L', 'Thalamus_R', 'Heschl_L', 'Heschl_R', 'Temporal_Sup_L', 'Temporal_Sup_R', 'Temporal_Pole_Sup_L', 'Temporal_Pole_Sup_R', 'Temporal_Mid_L', 'Temporal_Mid_R', 'Temporal_Pole_Mid_L', 'Temporal_Pole_Mid_R', 'Temporal_Inf_L', 'Temporal_Inf_R', 'Cerebelum_Crus1_L', 'Cerebelum_Crus1_R', 'Cerebelum_Crus2_L', 'Cerebelum_Crus2_R', 'Cerebelum_3_L', 'Cerebelum_3_R', 'Cerebelum_4_5_L', 'Cerebelum_4_5_R', 'Cerebelum_6_L', 'Cerebelum_6_R', 'Cerebelum_7b_L', 'Cerebelum_7b_R', 'Cerebelum_8_L', 'Cerebelum_8_R', 'Cerebelum_9_L', 'Cerebelum_9_R', 'Cerebelum_10_L', 'Cerebelum_10_R', 'Vermis_1_2', 'Vermis_3', 'Vermis_4_5', 'Vermis_6', 'Vermis_7', 'Vermis_8', 'Vermis_9', 'Vermis_10']
    harvard_oxford = set(cortical_regions) & set(labels_harvard_oxford)
    aal = set(cortical_regions) & set(labels_aal)
    if len(harvard_oxford)>0 and len(aal)>0:
        return 'both'
    elif len(harvard_oxford):
        return 'harvard_oxford'
    elif len(aal):
        return 'aal'
