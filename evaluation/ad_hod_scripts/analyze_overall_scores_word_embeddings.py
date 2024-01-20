import pandas as pd

def main():
    df = pd.read_csv(r'F:\dataset\predictions\overall\overview_predictions.csv')
    baseline_GloVe = 0.002532702616162057
    baseline_BERT = -0.0035220872142616455

    dfbert = df[df.embed_type=="BERT"]
    dfglove = df[df.embed_type=='GloVe']

    dfbert = dfbert[dfbert.cosine_similarity >= baseline_BERT]
    dfglove = dfglove[dfglove.cosine_similarity >= baseline_GloVe]



    # select unique above baseline regions
    bert_regions = dfbert.cortex_regions.unique()
    bert_regions.sort()
    l = [b for b in bert_regions]
    flattened_list = [item for sublist in bert_regions for item in sublist]

    bert_regions = [b for bl in bert_regions for b in bl]
    flattened_list = [item for sublist in array_of_lists for item in sublist]
    print('end')

if __name__ == '__main__':
    main()