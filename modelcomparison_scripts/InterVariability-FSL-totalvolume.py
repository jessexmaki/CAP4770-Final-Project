import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

concatenated_df = "" # Update this value to dataframe from parser


concatenated_df['Volume_mm3'] = pd.to_numeric(concatenated_df['Volume_mm3'], errors='coerce')
concatenated_df = concatenated_df.dropna(subset=['Volume_mm3'])
concatenated_df['Model'] = concatenated_df['Model'].astype('category')

# Compute the IQR for 'Volume_mm3'
Q1 = concatenated_df['Volume_mm3'].quantile(0.25)
Q3 = concatenated_df['Volume_mm3'].quantile(0.75)
IQR = Q3 - Q1

# Filter out the outliers by keeping only the valid values
concatenated_df = concatenated_df[~((concatenated_df['Volume_mm3'] < (Q1 - 1.5 * IQR)) | (concatenated_df['Volume_mm3'] > (Q3 + 1.5 * IQR)))]

plt.figure(figsize=(10, 8))
sns.set(style='whitegrid')

sns.boxplot(x='Model', y='Volume_mm3', data=concatenated_df, sym='')

plt.title('Inter-Model Variability FSL - Total Volume')
plt.xlabel('Model')
plt.ylabel('Total Volume')

plt.yscale('log')
plt.ylim(auto=True)

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()