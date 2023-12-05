import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict

from dataset_loader.dataset import LPPDataset
from dataset_loader.section_participant_base import BaseSectionParticipant
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from external_repos.nn_decoding.src import util
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
def convert_to_array(s):
    # Remove brackets, replace '\n' with space, split by space, and convert to float
    return np.array([float(x) for x in s.strip('[]').replace('\n', ' ').split()])

def main():
    dataset = LPPDataset()
    participant = 'sub-EN057'
    ps = BaseSectionParticipant(dataset[0])


    n_volumes = 150
    # get images
    subject_images = ps.fmri[...,:n_volumes].numpy()
    # do pca images?
    # test = torch.cat([ps[i] for i in range(5)], dim = 2)
    # test = torch.cat([ps[i] for i in range(5)], dim=-1)

    # pca = PCA(project).fit(subject_images)
    # mask images?

    # get words per volume
    words = [ps.get_words_volume_idx(i) for i in range(n_volumes)]
    no_words = [0]+[len(l) for l in words]
    index_words = np.cumsum(no_words)

    # load embeddings
    embed_type = 'GloVe'
    path = f'F:\\dataset\\annotation\\EN\\lppEN_word_embeddings_{embed_type}.csv'
    df = pd.read_csv(path)
    df[embed_type] = df[embed_type].apply(convert_to_array)

    # get embeddings of volumes: calculate mean if multiple words.
    # TODO: make this more neat, by actually extracting word embeeddings correctly
    encodings = np.array([df[embed_type].iloc[index_words[i]:index_words[i+1]].mean() for i in range(len(no_words)-1)])
    # encodings = [df.BERT.iloc[no_words[i]:no_words[i+1]].mean() for i in range(len(no_words)-1)]


    assert len(encodings)== subject_images.shape[3]

    # Final data prep: normalize.
    X = subject_images - subject_images.mean(axis=0)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    X=np.nan_to_num(X) #fill na
    Y = encodings - encodings.mean(axis=0)
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    # Y = Y.T



    # Prepare nested CV.
    # Inner CV is responsible for hyperparameter optimization;
    # # outer CV is responsible for prediction.
    # state = 42
    # n_folds = 3
    # ALPHAS = [1e-3]
    # inner_cv = KFold(n_splits=n_folds, shuffle=True, random_state=state)
    # outer_cv = KFold(n_splits=n_folds, shuffle=True, random_state=state)
    #
    # # Run inner CV.
    # gs = GridSearchCV(Ridge(fit_intercept=False,),
    #                   {"alpha": ALPHAS}, cv=inner_cv, verbose=10)
    # # Run outer CV.
    # decoder_predictions = cross_val_predict(gs, X, Y, cv=outer_cv)



    # new attempt ridge regression:

    # Assuming X is your fMRI data and Y is your labels
    X_shape = X.shape
    Y_shape = Y.shape

    # Reshape the fMRI data to 2D (flattening the spatial dimensions)
    X_reshaped = X.reshape((X_shape[0] * X_shape[1] * X_shape[2], X_shape[3])).T

    # Reshape the labels to 2D
    # Y_reshaped = Y

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_reshaped, Y, test_size=0.2, random_state=42)

    # Fit a Ridge Regression model
    alpha = 1.0  # You can adjust this regularization parameter
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train, Y_train)

    # Predict on the test set
    Y_pred = ridge_model.predict(X_test)

    # Evaluate the model (you can use different metrics depending on your task)
    mse = mean_squared_error(Y_test, Y_pred)
    print(f'Mean Squared Error: {mse}')
    mse = mean_squared_error(Y_test, Y_pred)

    # cosine similarity

    # Assuming Y_pred and Y_test have shape (30, 300)
    cosine_similarities = cosine_similarity(Y_pred, Y_test)
    from numpy import dot
    from numpy.linalg import norm
    a = Y_pred
    b = Y_test
    cos_sim = dot(a, b) / (norm(a) * norm(b))

    print('end')


if __name__ == '__main__':
    main()