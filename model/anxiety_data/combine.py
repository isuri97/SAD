import pandas as pd


df1 = pd.read_csv('output.csv')

df2 = pd.read_csv('output-an.csv')
combined_df = pd.concat([df1, df2], ignore_index=True)

# Save the combined dataframe as a CSV file
combined_df.to_csv('combined.csv', index=False)