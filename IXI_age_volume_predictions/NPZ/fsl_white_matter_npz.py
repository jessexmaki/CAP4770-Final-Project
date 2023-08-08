import os
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom

data_directory = 'PVE/0'
all_files = os.listdir(data_directory)

# Adjusted for the new filename format
subject_list = [filename.replace('.nii_seg_pve_0.nii.gz', '') for filename in all_files if filename.endswith('.nii_seg_pve_0.nii.gz')]

def extract_numeric_id(subject_id):
    return int(''.join(filter(str.isdigit, subject_id)))

df = pd.read_csv('IXI-FSL.csv')
df.drop_duplicates(subset='IXI_ID', keep='first', inplace=True)
demographic_dict = df.set_index('IXI_ID').to_dict(orient='index')

data_list = []
labels = []

def resize_volume(img):
    desired_depth = 128
    desired_width = 128
    desired_height = 128
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    img = zoom(img, (width_factor, height_factor, depth_factor))
    return img

for subject in subject_list:
    img = nib.load(os.path.join(data_directory, f"{subject}.nii_seg_pve_0.nii.gz"))  # Adjusted filename format
    img_data = resize_volume(img.get_fdata())
    img_data = img_data / np.max(img_data)
    subject_id = extract_numeric_id(subject)
    demographic_data = demographic_dict.get(subject_id)
    if demographic_data is None:
        print(f"Warning: No corresponding record found for image {subject}.nii_seg_pve_0.nii.gz")
        continue
    age = demographic_data['AGE']
    data_list.append(img_data)
    labels.append(age)
    print(f"Image {subject}.nii_seg_pve_0.nii.gz loaded successfully!")  # Adjusted print statement

labels = np.array(labels)
data_list = np.array(data_list)

x_temp, x_test, y_temp, y_test = train_test_split(data_list, labels, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.2, random_state=42)

np.savez('/content/drive/My Drive/Final-Project/preprocessed_data0.npz', x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test)
