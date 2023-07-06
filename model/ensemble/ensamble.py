import pandas as pd

# Read the three text files into dataframes
df1 = pd.read_csv('validation-result-xlnet.tsv', delimiter='\t')
df2 = pd.read_csv('validationsubmission3.tsv', delimiter='\t')
df3 = pd.read_csv('validationsubmission5.tsv', delimiter='\t')

# Concatenate the label columns from all three dataframes
merged_labels = pd.concat([df1['label'], df2['label'], df3['label']], axis=1)
mode_labels = merged_labels.mode(axis=1)[0]

result_df = pd.DataFrame({'id': df1['id'], 'label': mode_labels})

result_df.to_csv('output_file.tsv', index=False, sep='\t')