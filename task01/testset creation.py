import pandas as pd

val_df = pd.read_csv('validation.tsv', sep="\t")

new_df = val_df[['tweet_id', 'labels']].copy()
new_df.to_csv('test.tsv', sep='\t', index=False)
