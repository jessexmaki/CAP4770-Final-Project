import os

currentDir = os.getcwd()
fsDir = currentDir + '/freesurfer/'
manualDir = currentDir + '/manual/'
synthDir = currentDir + '/synthseg/synthseg_seg/'
outDir = currentDir + '/dice/'

fsCases = os.listdir(fsDir)
manualSegs = os.listdir(manualDir)
synthSegs = os.listdir(synthDir)

manualSegs.sort()

for manseg in manualSegs:
    caseID = manseg.split('_')
    if len(caseID) < 2:
        continue
    
    caseID = caseID[1]

    fsLabel = fsDir + caseID + '/mri/aseg.auto.mgz'
    synthLabel = synthDir + caseID + '_synthseg.nii'
    manLabel = manualDir + manseg

    synthCropPath = synthDir + 'crop/'
    manCropPath = manualDir + 'crop/'

    if not os.path.exists(synthCropPath):
        os.makedirs(synthCropPath)

    if not os.path.exists(manCropPath):
        os.makedirs(manCropPath)
    
    cropCmd = 'mri_convert --like ' + fsLabel + ' -rt nearest '
    diceCmd = 'mri_compute_overlap -a -s -l '

    synthCropLabel = synthCropPath + caseID + '_synthseg.nii'
    manCropLabel = manCropPath + caseID + '_manseg.nii'
    
    synthCropCmd = cropCmd + synthLabel + ' ' + synthCropLabel
    manCropCmd = cropCmd + manLabel + ' ' + manCropLabel

    print("Cropping cases on " + caseID)
    if not os.path.isfile(synthCropLabel):
        #print(synthCropCmd)
        os.system(synthCropCmd)
    else:
        print('Synth label found skipping crop...')

    if not os.path.isfile(manCropLabel):
        #print(manCropCmd)
        os.system(manCropCmd)
    else:
        print('Manual label found skipping crop...')

    fsOut = outDir + caseID + '_freesurfer_dice.csv'
    synthOut = outDir + caseID + '_synth_dice.csv'
    

    fsCmd = diceCmd + fsOut + ' ' + fsLabel + ' ' + manCropLabel
    synthCmd = diceCmd + synthOut + ' ' + synthCropLabel + ' ' + manCropLabel

    print("Calculating Dice on case " + caseID)
    if not os.path.isfile(fsOut):
        #print(fsCmd)
        os.system(fsCmd)
    if not os.path.isfile(synthOut):
        #print(synthCmd)
        os.system(synthCmd)
    

