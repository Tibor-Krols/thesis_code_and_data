import nilearn
from dataset_loader.dataset import LPPDataset
from dataset_loader.section_participant_base import BaseSectionParticipant
from nilearn import plotting

def main():
    dataset=LPPDataset()
    ps = BaseSectionParticipant(dataset[0])
    vol1= ps[0]
    vol2 = ps[1]

    img = nilearn.image.new_img_like(img, vol1, copy_header=True)

    # nilearn.plotting.plot_img_comparison()

    # plotting.plot_img(vol1)
    plotting.plot_img(img)
    plotting.show()



    filepath = r'F:\dataset\derivatives\sub-EN057\func\sub-EN057_task-lppEN_run-15_space-MNIColin27_desc-preproc_bold.nii.gz'

    masker.fit(filepath)
    plotting.plot_roi(
        masker.mask_img_, vol1, title="Mask from already masked data"
    )
    print('nd')
if __name__ == '__main__':
    main()