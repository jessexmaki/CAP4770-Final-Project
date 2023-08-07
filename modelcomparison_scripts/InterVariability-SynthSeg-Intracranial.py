import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

final_table_df = "" # update value to dataframe

# List of models (unique values in the 'Model' column)
final_table_df['Total Intracranial'] = pd.to_numeric(final_table_df['Total Intracranial'], errors='coerce')
final_table_df = final_table_df.dropna(subset=['Total Intracranial'])

final_table_df['Model'] = final_table_df['Model'].astype('category')


plt.figure(figsize=(10, 8))
sns.set(style='whitegrid')


sns.boxplot(x='Model', y='Total Intracranial', data=final_table_df, sym='')

plt.title('Inter-Model Variability - Total Intracranial')
plt.xlabel('Model')
plt.ylabel('Total Intracranial')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()