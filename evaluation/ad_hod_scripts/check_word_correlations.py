import nilearn
from dataset_loader.dataset import LPPDataset
from dataset_loader.section_participant_base import BaseSectionParticipant
from nilearn.input_data import NiftiMasker
import numpy as np
from nilearn import datasets, image, plotting
from nilearn.image import resample_img

def resample_mask(mask,vol):

    desired_shape = vol.shape
    desired_affine = np.eye(4)  # Replace this with the desired affine matrix
    desired_affine = vol.affine

    # Resample the mask to the desired shape and affine
    resampled_mask = resample_img(mask, target_shape=desired_shape, target_affine=desired_affine,interpolation='nearest')

    # Display the resampled mask
    plotting.plot_roi(resampled_mask, title=f'Prefrontal Cortex Mask Resampled to {desired_shape}')
    plotting.show()
    return resampled_mask



def get_destrieux_mask():

    # Load Destrieux atlas
    destrieux = datasets.fetch_atlas_destrieux_2009()

    # Assuming the prefrontal cortex is defined in the Destrieux atlas as index 14
    prefrontal_index_destrieux = 14

    # Extract the mask for the prefrontal cortex from the Destrieux atlas
    prefrontal_mask_destrieux = image.math_img('img == {}'.format(prefrontal_index_destrieux), img=destrieux['maps'])

    # Plot the prefrontal mask from Destrieux Atlas
    plotting.plot_roi(prefrontal_mask_destrieux, title='Prefrontal Cortex Mask (Destrieux)')
    plotting.show(    )
    return prefrontal_mask_destrieux

def get_oxford_mask(cortical_regions : list[str] = ['Temporal Pole']):

    # Load Harvard-Oxford cortical probabilistic atlas (50% threshold by default)
    harvard_oxford = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr50-1mm')

    # Assuming the prefrontal cortex is defined in the Harvard-Oxford atlas as index 12
    # prefrontal_index = harvard_oxford.labels.index('Temporal Pole')
    # prefrontal_index = harvard_oxford.labels.index('Temporal Pole')
    cortical_indices = [harvard_oxford.labels.index(region) for region in cortical_regions]

    # Extract the mask for the prefrontal cortex from the Harvard-Oxford atlas
    # cortical_mask = image.math_img('img == {}'.format(prefrontal_index), img=harvard_oxford.maps)
    cortical_mask = image.math_img('img == {}'.format(cortical_indices[0]), img=harvard_oxford.maps)
    for index in cortical_indices[1:]:
        prefrontal_mask = image.math_img('np.logical_or(img == {}, mask)'.format(index), img=harvard_oxford.maps,
                                         mask=cortical_mask)

    # Plot the prefrontal mask from Harvard-Oxford Atlas
    plotting.plot_roi(cortical_mask, title='Prefrontal Cortex Mask (Harvard-Oxford)')
    plotting.show()
    return cortical_mask
def get_aal_mask():

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



def main():
    dataset=LPPDataset(return_nii=True)
    sec = dataset[0]['cog_sequence']
    ps = BaseSectionParticipant(dataset[12],return_nii=True)
    vol1= ps[0]
    vol2 = ps[1]
    voln = ps[-1]

    img1= vol1
    img2 = vol2
    destrieux_mask = get_destrieux_mask()
    cortical_mask = get_oxford_mask()


    #mask volume
    # reshape mask
    mask = resample_mask(cortical_mask,vol1)
    # apply mask
    masked_volume = nilearn.masking.apply_mask([vol1,vol2,voln],mask_img=mask)

    masker = NiftiMasker(mask_img=prefrontal_mask)
    data1 = masker.fit_transform(img1)
    data2 = masker.fit_transform(img2)
    # img = nilearn.image.new_img_like(img, vol1, copy_header=True)

    # nilearn.plotting.plot_img_comparison()
    correlation = np.corrcoef(data1.T, data2.T)

    # plotting.plot_img(vol)
    print('nd')
if __name__ == '__main__':
    main()