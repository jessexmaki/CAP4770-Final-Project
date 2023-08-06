import pandas as pd

# vars
fileMetadata = {}
tableDataList = []
metadataCsvFile = 'fsl/FSL_csv/FSL_csv/FTHP_metadata.csv'

metadata_df = pd.read_csv(metadataCsvFile)
numRowsMetaData = len(metadata_df)


for rowNum in range (numRowsMetaData):
    csvFile = 'fsl/FSL_csv/FSL_csv/fast_{}_seg.csv'.format(rowNum)

    with open(csvFile, 'r') as file:
        tableData = []
        isTable = False
        for line in file:
            if line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    if parts[1] == "ColHeaders":
                        isTable = True
                        colHeaders = parts[2:]
                    elif not isTable:
                        fileMetadata[parts[1]] = ' '.join(parts[2:])
            elif isTable and line.strip():
                values = line.split()
                if len(values) == len(colHeaders):
                    tableData.append(values)

    if tableData:
        df = pd.DataFrame(tableData, columns=colHeaders)

        manufacturer_info = metadata_df.iloc[rowNum]['manufacturer']
        model_info = metadata_df.iloc[rowNum]['model']
        df['Manufacturer'] = manufacturer_info
        df['Model'] = model_info

        print(f"Data from file {csvFile}:")
        print(df)

        tableDataList.append(df)
    else:
        print(f"No table data found in the file {csvFile}.")

if tableDataList:
    concatenated_df = pd.concat(tableDataList, ignore_index=True)
    print("\nConcatenated DataFrame:")
    print(concatenated_df)