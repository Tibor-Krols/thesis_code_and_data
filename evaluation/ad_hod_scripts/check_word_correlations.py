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

def plot_timeseries(ps,nvolumes = 10):
    """ps needs to return nifti image"""
    for i in range(nvolumes):
        voli = ps[i]
        plotting.plot_roi(voli, title=f'vol {i}')
        plotting.show()


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
    # for index in cortical_indices[1:]:
    #     cortical_mask = image.math_img('np.logical_or(img == {}, mask)'.format(index), img=harvard_oxford.maps,
    #                                      mask=cortical_mask)

    cortical_mask = resample_mask(cortical_mask)
    # Plot the prefrontal mask from Harvard-Oxford Atlas
    plotting.plot_roi(cortical_mask, title=str(cortical_regions))
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


def compare_word_with_average_ps(ps,w1:str,w2:str,avg_fmri_w1,cortical_mask):
    # get fmri volumes containing w1
    word1_indices = ps.get_vol_idx_word(w1)
    word1_indices = sum(word1_indices,[])
    w1_fmri_volumes = [ps[i] for i in word1_indices]


    # get fmri volumes containing w2
    word2_indices = ps.get_vol_idx_word(w2)
    word2_indices = sum(word2_indices,[])
    w2_fmri_volumes = [ps[i] for i in word2_indices]

    # get average fmri of w1
    # assert w1 in avg_word_dict.keys()
    # avg_fmri_w1 = avg_word_dict[w1]
    # convert average to nfti format
    avg_fmri_w1 = make_nifti_image_from_tensor(avg_fmri_w1)



    # mask cortical region:
    # for w1
    w1_fmri_volumes = nilearn.masking.apply_mask(w1_fmri_volumes, mask_img=cortical_mask)
    # w1_fmri_volumes = resample_mask(w1_fmri_volumes)
    # for w2
    w2_fmri_volumes = nilearn.masking.apply_mask(w2_fmri_volumes, mask_img=cortical_mask)
    # w2_fmri_volumes = resample_mask(w2_fmri_volumes)
    # for average
    avg_fmri_w1 = nilearn.masking.apply_mask([avg_fmri_w1], mask_img=cortical_mask)



    # calculate correlation with average fmri of w1
    # for w1
    correlation_w1 = np.corrcoef(np.vstack((avg_fmri_w1,w1_fmri_volumes)))
    # for w2
    correlation_w2 = np.corrcoef(np.vstack((
        avg_fmri_w1,
        w2_fmri_volumes)))

    avg_cor_w1 = correlation_w1[0][1:].mean()
    avg_cor_w2 = correlation_w2[0][1:].mean()
    print(f'Average correlation of volumes from w1 and w2 with the average brain activity w1\n'
          f'w1 = {w1}, w2 = {w2}\n'
          f'w1 = {avg_cor_w1}\n'
          f'w2 = {avg_cor_w2}\n'
          f'avg_cor_w1 > avg_cor_w2 = {avg_cor_w1 > avg_cor_w2}\n'
          f'difference = {avg_cor_w1 - avg_cor_w2}')
    pass


def main():
    dataset=LPPDataset(return_nii=True)


    # sec = dataset[0]['cog_sequence']
    ps = BaseSectionParticipant(dataset[7],return_nii=True)
    avg_fmri_word_dict = load_averages_participant(ps.participant)

    # ps = BaseSectionParticipant(dataset[12],return_nii=True)
    # ps2 = BaseSectionParticipant(dataset[22],return_nii=True)
    cortex_region = 'Superior Temporal Gyrus, anterior division'
    cortical_mask = get_oxford_mask(cortical_regions= [cortex_region])

    word1 = 'drink'
    word2 = 'loved'
    compare_word_with_average_ps(
        ps,
        w1 = word1,
        w2 = word2,
        avg_fmri_w1=avg_fmri_word_dict[word1],
        cortical_mask=cortical_mask)








    word1_indices = ps.get_vol_idx_word(word1)
    word1_indices = sum(word1_indices,[])
    word2_indices = ps.get_vol_idx_word(word2)
    word2_indices = sum(word2_indices,[])
    vol_idx_list = word1_indices + word2_indices
    # vol_idx_list = sum(vol_idx_list,[])
    vol_list = [ps[i] for i in vol_idx_list]
    # vol_list = [ps2[word1_indices[0]]] + vol_list
    # vol_list1 = [ps[i] for i in range(1,100,10)]
    # vol_list2 = [ps2[i]for i in range(1,100,10)]
    # vol_list = vol_list1 + vol_list2
    vol1= ps[0]
    # vol2 = ps[1]
    # voln = ps[-1]

    # img1= vol1
    # img2 = vol2
    # destrieux_mask = get_destrieux_mask()
    # cortical_regions= ['Superior Temporal Gyrus, anterior division','Superior Temporal Gyrus, posterior division','Frontal Pole','Temporal Pole']
    cortical_regions= ['Superior Temporal Gyrus, anterior division','Frontal Pole','Temporal Pole']

    cortex_correlation = {}
    for cor_reg in cortical_regions:
        # get mask for cortical region
        cortical_mask = get_oxford_mask(cortical_regions= [cor_reg])

        #mask volume
        # reshape mask
        cortical_mask = resample_mask(cortical_mask)
        # apply mask
        masked_volumes = nilearn.masking.apply_mask(vol_list,mask_img=cortical_mask)
        correlation = np.corrcoef(masked_volumes)
        cortex_correlation[cor_reg] = correlation
        # print(cor_reg)

        # look at correlation
        word_index = 0
        # if word_index <= len(word1_indices)
        print(cor_reg)
        correlation_dropped_word = np.delete(correlation[word_index],[word_index])
        correlation_word1 = correlation_dropped_word[:len(word1_indices)-1]
        correlation_word2 = correlation_dropped_word[len(word1_indices)-1:]
        # print(f'correlation with {word1} ',correlation[word_index][:len(word1_indices)].mean())
        # print(f'correlation with {word2}',correlation[word_index][len(word1_indices):].mean())
        print(f'correlation with {word1} ',correlation_word1.mean())
        print(f'correlation with {word2}',correlation_word2.mean())

    # masker = NiftiMasker(mask_img=mask)
    # data1 = masker.fit_transform(img1)
    # data2 = masker.fit_transform(img2)
    # img = nilearn.image.new_img_like(img, vol1, copy_header=True)

    # nilearn.plotting.plot_img_comparison()
    correlation = np.corrcoef(masked_volumes)

    # plotting.plot_img(vol)
    print('nd')
if __name__ == '__main__':
    main()