import os
import matplotlib.pyplot as plt
import numpy as np

def addEntry(dir, file, dataMap):
    with open(dir + file) as data:
        for row in data:
            rowData = row.split('\t')
            labelID = int(rowData[0])
            dice = float(rowData[2])
            
            if labelID in dataMap:
                dataMap[labelID].append(dice)
            else:
                dataMap[labelID] = [dice]



diceDir = os.getcwd() + '/../dice/'
labelKeyFile = diceDir + 'fslabel_key.txt'

fsScoresMap = {}
synthScoresMap = {}
key = {}
numLabels = 0

for file in os.listdir(diceDir):
    if file.endswith('freesurfer_dice.csv'):
        addEntry(diceDir, file, fsScoresMap)
    elif file.endswith('synth_dice.csv'):
        addEntry(diceDir, file, synthScoresMap)


with open(labelKeyFile) as labelKey:
    for row in labelKey:
        rowData = row.split(' ')
        if len(rowData) < 2:
            continue
        number = rowData[0]
        name = rowData[1]
        key[int(number)] = name
        numLabels += 1

fsDice = []
synthDice = []
ticks = []

for label in key:
    ticks.append(key[label])
    synthDice.append(synthScoresMap[label])
    fsDice.append(fsScoresMap[label])

fig, ax = plt.subplots()

positions1 = []
positions2 = []

for post in range(1, len(key)*3 + 1):
    mod = post % 3
    if mod == 1:
        positions1.append(post)
    elif mod == 2:
        positions2.append(post)


bplot1 = ax.boxplot(synthDice, positions=positions1, labels=ticks, patch_artist=True, boxprops=dict(facecolor="pink"))
bplot2 = ax.boxplot(fsDice, positions=positions2, labels=ticks, patch_artist=True, boxprops=dict(facecolor="lightblue"))

legendLabels = ['SynthSeg', 'FreeSurfer']
ax.legend([bplot1['boxes'][0], bplot2['boxes'][0]], legendLabels, loc='upper right')
ax.set_xticks(np.array(positions1)+0.5)

ax.set_ylabel('Dice Score')
ax.set_title('Dice score comparison of SynthSeg and FreeSurfer')
plt.xticks(rotation = 90)
plt.show()