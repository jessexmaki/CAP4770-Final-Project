import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

final_table_df = "" # Update to dataframe from parser

final_table_df['Total Intracranial'] = pd.to_numeric(final_table_df['Total Intracranial'], errors='coerce')
final_table_df = final_table_df.dropna(subset=['Total Intracranial'])
final_table_df['Model'] = final_table_df['Model'].astype('category')

# Compute the IQR for 'Total Intracranial'
Q1 = final_table_df['Total Intracranial'].quantile(0.25)
Q3 = final_table_df['Total Intracranial'].quantile(0.75)
IQR = Q3 - Q1

# Filter out the outliers by keeping only the valid values
final_table_df = final_table_df[~((final_table_df['Total Intracranial'] < (Q1 - 1.5 * IQR)) | (final_table_df['Total Intracranial'] > (Q3 + 1.5 * IQR)))]

plt.figure(figsize=(10, 8))
sns.set(style='whitegrid')

sns.boxplot(x='Model', y='Total Intracranial', data=final_table_df, sym='')

plt.title('Inter-Model Variability SynthSeg - Total Volume')
plt.xlabel('Model')
plt.ylabel('Total Volume')

# Set the y-axis to a logarithmic scale
plt.yscale('log')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()