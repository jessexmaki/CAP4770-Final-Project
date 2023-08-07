import os
import numpy as np
import nibabel as nib
import csv
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

%matplotlib inline

def extract_numeric_id(subject_id):
    return int(''.join(filter(str.isdigit, subject_id)))

def calculate_and_add_volume(data_directory, pve_suffix, volume_name):
    # Get list of subject IDs
    all_files = os.listdir(data_directory)
    subject_list = [filename.replace(pve_suffix, '') for filename in all_files if filename.endswith(pve_suffix)]

    # Calculate volumes
    volumes = []
    for subject in subject_list:
        pve_file = os.path.join(data_directory, f'{subject}{pve_suffix}')

        # Load PVE file
        pve_data = nib.load(pve_file).get_fdata()

        # Calculate volume
        volume = np.sum(pve_data)

        # Append to list
        volumes.append(volume)

    # Read CSV file
    csv_file_path = 'IXI-FSL.csv'
    csv_data = []

    with open(csv_file_path, 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            csv_data.append(row)

    # Add volumes to CSV
    for i, subject_id in enumerate(subject_list):
        numeric_subject_id = extract_numeric_id(subject_id)

        for row in csv_data:
            if 'IXI_ID' in row and int(row['IXI_ID']) == numeric_subject_id:
                row[volume_name] = volumes[i]
                break

    # Write updated CSV
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = list(csv_data[0].keys())
        if volume_name not in fieldnames:
            fieldnames.append(volume_name)

        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(csv_data)

    print(f"{volume_name} volumes added to the CSV file.")

# Set data directories and calculate volumes for each PVE type
calculate_and_add_volume('PVE/0', '.nii_seg_pve_0.nii.gz', 'WM_VOLUME')
calculate_and_add_volume('PVE/1', '.nii_seg_pve_1.nii.gz', 'GM_VOLUME')
calculate_and_add_volume('PVE/2', '.nii_seg_pve_2.nii.gz', 'CSF_VOLUME')
