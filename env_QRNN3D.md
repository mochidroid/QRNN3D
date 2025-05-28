# Setting Up the Environment for QRNN3D

## 🛠 Step 1: Install WSL2 and Ubuntu

1. Open PowerShell as Administrator and run:
    ```powershell
    wsl --install
    ```
    > *If WSL2 and Ubuntu are already installed, you can skip this step.*

2. Verify that WSL is running version 2:
    ```powershell
    wsl --list --verbose
    ```
    Ensure that your Ubuntu instance is listed as version 2.

3. Install Ubuntu:
    ```powershell
    wsl --install -d Ubuntu
    ```

4. Set username and password


## 🐍 Step 2: Update Ubuntu and Install Miniconda

1. **Update package lists and upgrade existing packages:**
    ```bash
    sudo apt update && sudo apt upgrade -y
    ```

2. **Install `wget` and `git`:**
    ```bash
    sudo apt install wget git -y
    ```

3. **Download and run the Miniconda installer:**
    ```bash
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
    ```
    Follow the prompts to complete the Miniconda installation.

4. **Reload your shell to apply changes:**
    ```bash
    source ~/.bashrc
    ```

5. **Verify that Conda is installed correctly:**
    ```bash
    conda --version
    ```

## 🧪 Step 3: Set Up the QRNN3D Virtual Environment

1. **Create and activate a Conda environment for QRNN3D:**
    ```bash
    conda create -n qrnn3d_env python=3.6
    conda activate qrnn3d_env
    ```

2. **Install Caffe and dependencies:**
    ```bash
    conda install -c conda-forge caffe
    ```
    > *After installation, you can verify Caffe's Python bindings with:*
    > ```bash
    > python -c 'import caffe; print("Caffe imported successfully!")'
    > ```
    > *If you see `Caffe imported successfully!`, Caffe is set up correctly.*

    ```bash
    conda install numpy scipy scikit-image matplotlib tqdm h5py
    ```

3. **Clone the QRNN3D repository:**
    ```bash
    cd ~  # Move to your home directory
    git clone https://github.com/mochidroid/QRNN3D.git
    cd QRNN3D
    ```

4. **Install PyTorch (with CUDA support if available):**
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
    > *After installation, check PyTorch and CUDA availability with:*
    > ```bash
    > python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
    > ```

5. **Install torchnet and tensorboardX:**
    ```bash
    pip install torchnet tensorboardX lmdb
    ```

    ---

## ✅ Setup Complete

The environment setup is complete.  
For instructions on how to run the project, please refer to `run_command.txt` in the project directory.

