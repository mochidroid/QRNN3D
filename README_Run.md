# QRNN3D: Hyperspectral Image Denoising

This repository provides the implementation and execution scripts for **QRNN3D**,  
a deep learning–based method for hyperspectral image denoising.

---

## 📂 Input Data

- Input hyperspectral data should be **normalized to the range [0, 1]**.
- The datasets and checkpoints are located in:
`MATLAB_Share/Data_QRNN3D/`

---

## 🚀 How to Run

To start the denoising process, simply execute the following shell script:

```bash
./run_all.sh

```

This script will:
- Load the necessary pretrained models and configuration files.
- Perform denoising on all specified datasets.
- Save the results automatically in the appropriate result directories.


## ⚙️ Environment Setup

For environment installation and dependencies, please refer to:

`env_QRNN3D.md`


Author:
Shingo Takemoto (Institute of Science Tokyo)
Project: Hyperspectral Image Denoising using QRNN3D
