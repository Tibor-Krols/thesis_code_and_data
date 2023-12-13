import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# IN what list to place it?
# all volumes embeddings? (+-2800 cases. )

# determine volumes that were predicted (test set)

# 1. calculate mean embeddings per volume (for each section)


# for each predicted volume:
#     calculate similarity with all other volumes

    # sort similarities
    # find rank of actual volume embedding.


def get_rank_accuracy(pred, embed_list):
    pass


def pairwise_comparison(pred1, pred2, gt1, gt2):
    # cs = cosine_similarity([pred1,pred2,gt1,gt2],[pred1,pred2,gt1,gt2])
    cs = cosine_similarity([pred1,pred2],[gt1,gt2])

    better = (
        cs[0,0]>cs[0,1]
             ) and (
        cs[1,1]>cs[1,0]
    )
    return better

def main():

    pred1 = np.array([1,1,0,0])
    gt1 = np.array([1,2,0,0])
    pred2 = np.array([0,0,1,2])
    gt2 = np.array([0,0,0,1])

    better = pairwise_comparison(pred1=pred1,
                        pred2=pred2,
                        gt1=gt1,
                        gt2=gt2)
    print('done')
if __name__ == '__main__':
    main()