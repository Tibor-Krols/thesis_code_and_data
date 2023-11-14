import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from models.bayesian.fmri_averages_per_participant import load_averages_participant
from utils.cortical_masking import get_oxford_mask, mask_avg_fmri_word_dict



participant = 'sub-EN057'
avg_fmri_word_dict = load_averages_participant(participant)
cortical_regions = ['Superior Temporal Gyrus, anterior division']
cortical_mask = get_oxford_mask(cortical_regions=cortical_regions)
# mask avg fmri images if mas provided:
masked_avg_fmri_word_dict = mask_avg_fmri_word_dict(avg_fmri_word_dict, cortical_mask)


# Extract the words and vectors
words = list(masked_avg_fmri_word_dict.keys())
vectors = np.array(list(masked_avg_fmri_word_dict.values()))

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

# Create a scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c='blue', edgecolors='k')

# Annotate each point with its word
for word, (x, y) in zip(words, vectors_2d):
    plt.annotate(word, (x, y), fontsize=8, ha='right')

plt.title('PCA Visualization of Word Embeddings')
plt.show()