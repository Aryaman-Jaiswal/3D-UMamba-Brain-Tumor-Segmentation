# 3D-UMamba-Brain-Tumor-Segmentation

## Overview

This repository contains a fully 3D volumetric brain tumor segmentation pipeline utilizing the **U-Mamba architecture**. Developed as a robust alternative to traditional Vision Transformers (ViTs) and CNNs, this system accurately delineates highly aggressive brain tumors (gliomas) from multi-modal MRI scans.

By leveraging continuous-time State Space Models (SSMs), the U-Mamba network achieves the infinite global receptive field of a Transformer while maintaining linear computational complexity. This allows the model to process high-resolution 3D medical data in its original form without the severe memory limitations that often force researchers to slice 3D volumes into 2D images.

Link to Kaggle notebook: https://www.kaggle.com/code/aryamanjaiswal001/brain-tumor-project/edit

## Key Features

* **3D Volumetric Processing:** Replaces 2D slice-by-slice processing to fully utilize Z-axis depth information and spatial continuity.


* **Linear-Time Sequence Modeling:** Uses a Vision Mamba bottleneck to capture long-range global dependencies efficiently using hardware-aware parallel prefix scans.


* **Custom Morphological Boundary Loss:** Implements a pure PyTorch-based boundary loss function (via 3D Max Pooling) combined with Dice Cross-Entropy. This bypasses heavy C++ compilation constraints on cloud GPUs while forcing the network to learn intricate, spiky tumor margins.


* **Algorithmic False Positive Suppression:** Utilizes a Size-Aware Connected Component Analysis filter during post-processing to isolate the main tumor mass and eliminate isolated "phantom tumor" artifacts.


* **Sliding Window Inference:** Smoothly merges overlapping 3D patches across unseen patient volumes using Gaussian blending.



## Dataset

The model is trained and evaluated on the **BraTS 2020** dataset.

* **Inputs:** Multi-modal 3D NIfTI volumes (T1, T1ce, T2, FLAIR).


* **Targets:** Multi-class sub-region segmentation:
* Whole Tumor (WT) 


* Tumor Core (TC) 


* Enhancing Tumor (ET) 





## Architecture Details

The network is a well-optimized, U-shaped encoder-decoder composition:

1. **Encoder:** Standard 3D Residual Convolutional Blocks extract high-frequency local textures (e.g., micro-vascular proliferation).


2. **Bottleneck:** A 3D Vision Mamba block flattens low-resolution 3D feature maps into a 1D sequence, applies a selective state space mechanism to map global dependencies, and reshapes the tensor back to a 3D volume.


3. **Decoder:** 3D Upsampling blocks utilize skip connections to restore exact anatomical borders during reconstruction.



## Results

The model was tested on a 73-patient validation dataset and achieved a **Mean Dice Similarity Coefficient (DSC) of 81.58%** after just 75 epochs of training.

Visual analysis confirms that the extended training alongside the connected component post-processing filter successfully separated the main glioma mass and removed healthy-tissue false positives.


*<img width="799" height="738" alt="Screenshot 2026-06-28 003357" src="https://github.com/user-attachments/assets/775b549f-bba0-4e88-ac63-419b10afd84f" />*

## Tech Stack

* **Deep Learning Framework:** PyTorch 2.6.0 (CUDA 12.1) 


* **Medical Imaging:** MONAI 1.4.0 


* **State Space Models:** `mamba-ssm` 2.2.3, `causal-conv1d` 1.4.0 


* **Environment:** Kaggle Interactive Notebooks (Dual NVIDIA Tesla T4 GPUs) 



## Author

**Aryaman Jaiswal** Vellore Institute of Technology (VIT)
