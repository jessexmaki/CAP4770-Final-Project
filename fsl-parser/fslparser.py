import pandas as pd

# vars
fileMetadata = {}
tableData = []
isTable = False
csvFile = 'fsl/FSL_csv/FSL_csv/fast_0_seg.csv'

with open(csvFile, 'r') as file:
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
    print(df)
else:
    print("No table data found in the file.")