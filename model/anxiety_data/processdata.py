import pandas as pd
import glob

# List all CSV files in the directory
file_list = glob.glob('*.csv')

# Initialize empty lists to store post values and labels
post_values = []
labels = []

# Iterate through each CSV file
for file in file_list:
    df = pd.read_csv(file)  # Read the CSV file
    post_column = df['post']  # Extract the 'post' column values
    post_values.extend(post_column)  # Add the values to the list
    labels.extend([0] * len(post_column))  # Add 1 as the label for each row

# Create a new dataframe with the extracted post values and labels
new_df = pd.DataFrame({'post': post_values, 'label': labels})

# Write the new dataframe to a CSV file
new_df.to_csv('output.csv', index=False)