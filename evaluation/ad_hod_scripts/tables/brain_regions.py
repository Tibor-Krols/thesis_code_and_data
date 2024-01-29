import pandas as pd



df = pd.read_excel(r'C:\Users\tibor\Documents\thesis\report\tables\brain_regions.xlsx')
table = df.to_latex(index=False)
print(table)
print('done')