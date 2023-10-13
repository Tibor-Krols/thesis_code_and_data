import os
def save_df(df,filename,save_path):
    if not filename.endswith('.csv'):
        filename +='.csv'
    os.makedirs(save_path,exist_ok=True)
    df.to_csv(os.path.join(save_path,filename),index=False)
    print(f"saved {filename} to {save_path}")
