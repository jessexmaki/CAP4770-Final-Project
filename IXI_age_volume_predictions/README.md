The directory contains pre-processing scripts and files specifically tailored for age-volume machine learning and deep learning predictions using the IXI Dataset.

1. FSL_matter_calculations:
- calculations.py: A script for relevant calculations.
-Google Drive Link: Provides access to FSL-segmented images, showcasing white matter, gray matter, and CSF volumes.

2. NPZ:
- Source Code: Used to compress images for the 3D CNN block.
- Models Parameters: Houses parameters for both X and Y models, covering training, testing, and other related aspects.
- Google Drive Link: Provides access to .npz files corresponding to FSL white matter, FSL gray matter, and SynthSeg segmentations.
csv:

3. Contains CSV files essential for associating images with corresponding ages and IXI IDs, a pivotal step for age-volume predictions.

4. Age_Volume_Predictions.ipynb:
- Features the source code for the machine learning model.
- Includes plots that illustrate the comparison between truth and prediction values.
