# Vault-Prediction

This program predicts the expected vault size after implantable collamer lens (ICL) implantation, given pre-operative ultrasound biomicroscopy (UBM) and ICL size. Input data includes excel sheet that contains patients' information including the size of ICL used in the operation and pre-operative UBM images. The model architecture is based on resnet, with an additional layer that takes into account ICL sizes.

Directory Structure:
Vault-Prediction
 - resnet.py
 - data/
     - <ICL sizes (excel)>
     - UBM
         - Subject 1
         - ..
 - model_weights/

Current MAE So far (Post-Operative vault size ranging 200 ~ 900) :
~ 140

# Vault-Prediction

Vault-Prediction is a program that predicts the expected vault size after implantable collamer lens (ICL) implantation. It takes pre-operative ultrasound biomicroscopy (UBM) data and the size of the ICL used in the operation as input. The model architecture is based on ResNet, with an additional layer that incorporates ICL sizes for more accurate predictions.

## Directory Structure
Vault-Prediction/
│
├── resnet.py
│
├── data/
│ ├── <ICL_sizes.xlsx> # Excel sheet containing ICL sizes
│ └── UBM/
│ ├──     Subject 1/
│ ├── ... # Subdirectories containing UBM images
│
└── model_weights/

## Current Mean Absolute Error (MAE)

The current Mean Absolute Error (MAE) for predicting post-operative vault size, which ranges from 200 to 900, is approximately 140.

