import os

currentDir = os.getcwd()
fsSubjDir = currentDir + '/freesurfer/'
files = os.listdir(currentDir)
files.sort()
files.reverse()
# export SUBJECTS_DIR=$(pwd)/freesurfer

for file in files:
    id = (file.split('_'))
    if len(id) < 2:
        continue
    
    if not file.endswith('.nii.gz'):
        continue

    id = id[1]
    pathIn = currentDir + '/' + file
    outName = id + '_synthseg.nii'
    outCsv = id + '_synthseg_vol.csv'
    pathOut = currentDir + '/synthseg/synthseg_seg/' + outName
    pathCsv = currentDir + '/synthseg/synthseg_csv/' + outCsv

    segcmd = 'mri_synthseg --i ' + pathIn \
         + ' --o ' + pathOut \
         + ' --vol ' + pathCsv

    fsCmd1 = 'recon-all -s ' + id + ' -i ' + file + ' -autorecon1'
    fsCmd2 = 'recon-all -subcortseg -subjid ' + id

    outfile = fsSubjDir + id
    if os.path.isdir(outfile):
        continue

    print('Running: ' + id + '...')
    os.system(segcmd)
    os.system(fsCmd1)
    os.system(fsCmd2)
    print(segcmd)
    #print(fsCmd1)
    #print(fsCmd2)
    #print(segcmd)
