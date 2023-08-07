import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#update to apply dataframe from parser
concatenated_df = "dataframe"

concatenated_df['NVoxels'] = pd.to_numeric(concatenated_df['NVoxels'], errors='coerce')
concatenated_df = concatenated_df.dropna(subset=['NVoxels'])

concatenated_df['Model'] = concatenated_df['Model'].astype('category')


plt.figure(figsize=(10, 8))
sns.set(style='whitegrid')


sns.boxplot(x='Model', y='NVoxels', data=concatenated_df, sym='')

plt.title('Inter-Model Variability - NVoxels')
plt.xlabel('Model')
plt.ylabel('NVoxels')

# Set the y-axis to a logarithmic scale
plt.yscale('log')  

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()