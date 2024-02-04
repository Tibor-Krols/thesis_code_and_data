import pandas as pd

def main():
    dfbase = pd.read_csv(r'C:\Users\tibor\Documents\thesis\code_and_data\evaluation\metrics_backup\baseline_metrics_bertScore_no_fixed_randomstate.csv')



    df57 = pd.read_csv(r'C:\Users\tibor\Documents\thesis\code_and_data\evaluation\metrics\bayesian\per_participant\bayesian_metrics_per_participant_57.csv')
    df58 = pd.read_csv(r'C:\Users\tibor\Documents\thesis\code_and_data\evaluation\metrics\bayesian\per_participant\bayesian_metrics_per_participant_58.csv')

    cols = ['rouge1', 'rouge2', 'rougeL', 'bleu', 'meteor', 'wer', 'bert_score']

    df57[cols].mean().tolist()

    # Create a dictionary to hold the metrics for each participant
    data = {'sub-EN57':  df57[cols].mean().tolist(),
            'sub-EN58':  df58[cols].mean().tolist(),
            'Baseline':  dfbase[cols].mean().tolist()}

    # Create a DataFrame from the dictionary with metric names as columns
    df = pd.DataFrame(data, index=cols).T

    latex_string = df.to_latex()
    latex_string = latex_string.replace('_','\\_')
    print(latex_string)
    print('done')

if __name__ == '__main__':
    main()