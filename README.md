# Vault-Prediction

Vault-Prediction is a program that predicts the expected vault size after implantable collamer lens (ICL) implantation. It takes pre-operative ultrasound biomicroscopy (UBM) data and the size of the ICL used in the operation as input. The model architecture is based on ResNet, with an additional layer that incorporates ICL sizes for more accurate predictions.

## Directory Structure
```plaintext
Vault-Prediction/
├── resnet.py
├── data/
│   ├── ICL_sizes.xlsx  # Excel sheet containing ICL sizes
│   └── UBM/
│       ├── Subject 1/  # Subdirectories containing individual UBM images
│       ├── Subject 2/
│       ├── ...         
└── model_weights/      # Learned model weights saved in
```

## Current Mean Absolute Error (MAE)

The current Mean Absolute Error (MAE) for predicting post-operative vault size, which ranges from 200 to 900, is approximately 140.
