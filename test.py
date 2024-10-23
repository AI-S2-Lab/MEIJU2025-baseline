import pandas as pd

# Load the CSV file
file_path = 'submission.csv'
df = pd.read_csv(file_path)

# Replace 'emo_pred' and 'int_pred' columns' values with 'neutral'
df['emo_pred'] = 'neutral'
df['int_pred'] = 'neutral'

# Save the updated dataframe to a new CSV file
output_path = 'submission.csv'
df.to_csv(output_path, index=False)
