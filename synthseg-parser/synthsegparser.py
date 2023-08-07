import pandas as pd

# vars
metadataCsvFile = 'synthseg_csv/synthseg_csv/FTHP_metadata.csv' # Update path if necessary
metadata_df = pd.read_csv(metadataCsvFile)
tableDataList = []

# Iterate over the metadata rows
for index, row in metadata_df.iterrows():
    scanId = row['scanID']
    manufacturer = row['manufacturer']
    model = row['model']

    csvFile = 'synthseg_csv/synthseg_csv/{}_synthseg_vol.csv'.format(scanId) # Update path if necessary
    
    # Check if the file exists
    try:
        # Reading the CSV file into a DataFrame
        df = pd.read_csv(csvFile)

        # Check if 'total intracranial' column has any non-null value
        if df['total intracranial'].isnull().all():
            print(f"Skipping file {csvFile} as it contains no value in the 'total intracranial' column.")
            continue

        # Extracting the 'total intracranial' column value
        total_intracranial_value = df['total intracranial'].values[0]

        tableDataList.append({
            'Scan ID': scanId,
            'Total Intracranial': total_intracranial_value,
            'Manufacturer': manufacturer,
            'Model': model
        })

    except FileNotFoundError:
        print(f"File {csvFile} not found.")
        continue

final_table_df = pd.DataFrame(tableDataList)
print("\nDataFrame with Total Intracranial data, Manufacturer, and Model:")
print(final_table_df)