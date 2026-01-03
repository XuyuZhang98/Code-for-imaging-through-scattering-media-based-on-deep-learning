**Author: Xuyu Zhang

This software package contains the source code for the manuscript "Deep Learning for Imaging through Scattering Media". It implements two deep learning models (U-Net and Conditional GAN) to reconstruct object images from speckle patterns.

--------------------------------------------------------------------------------
1. SYSTEM REQUIREMENTS
--------------------------------------------------------------------------------

### Operating Systems
The software has been tested on the following operating systems:
* **Windows:** Windows 10 and Windows 11 (64-bit)
* **Linux:** Ubuntu 20.04 LTS (Tested environment)

### Software Dependencies
The code requires **Python 3.8 or higher**. The specific library dependencies and tested versions are listed below:

* **TensorFlow** (Tested on v2.10.0 and v2.4.0; requires >= 2.4.0)
* **NumPy** (Tested on v1.23.5; requires >= 1.19.0)
* **Matplotlib** (Tested on v3.7.0; requires >= 3.3.0)
* **Pillow (PIL)** (Tested on v9.4.0; requires >= 8.0.0)
* **Tkinter** (Standard Python GUI library, usually included with Python)
* **Tqdm** (Tested on v4.64.0)

### Hardware Requirements
* **Non-standard Hardware:** A CUDA-enabled **NVIDIA GPU** is highly recommended for training and fast inference.
* **Tested Hardware:** NVIDIA GeForce RTX 3090 (24GB VRAM).
* **Minimum Requirements:** NVIDIA GPU with at least 8GB VRAM (for GAN training). CPU-only execution is possible for inference (Demo) but not recommended for training.

--------------------------------------------------------------------------------
2. INSTALLATION GUIDE
--------------------------------------------------------------------------------

### Instructions
1.  **Install Python:** Ensure Python 3.8+ is installed on your system.
2.  **Download Code:** Extract the submitted code folder to your local machine.
3.  **Install Dependencies:** Open a terminal or command prompt in the code directory and run the following command to install all required libraries:

    ```bash
    pip install tensorflow numpy matplotlib pillow tqdm
    ```

    *(Note: For GPU acceleration, please ensure that the NVIDIA CUDA Toolkit and cuDNN library compatible with your TensorFlow version are correctly installed on your system.)*

### Typical Install Time
On a "normal" desktop computer with a standard broadband internet connection, the installation of dependencies typically takes **5 to 10 minutes**.

--------------------------------------------------------------------------------
3. DEMO
--------------------------------------------------------------------------------

We provide Graphical User Interface (GUI) tools to demonstrate the reconstruction capabilities without running the full training process.

### Instructions to Run on Data
1.  Navigate to the `U-Net` or `GAN` folder.
2.  Run the inference script:
    * **U-Net Demo:** `python unet_reconstruction_demo.py`
    * **GAN Demo:** `python Gan_reconstruction_demo.py`
3.  **In the GUI:**
    * Click "Browse..." to load the provided pre-trained weights (e.g., `best_model.h5` or `.keras`).
    * Click "Add Images" to select sample speckle patterns (provided in `dataset/input/`).
    * Select an output folder and click "Start Reconstruction".

### Expected Output
The software will process the input speckle patterns and save the reconstructed images (Grayscale or Binary format) to the selected output directory. The reconstructed images should visually resemble the ground truth objects (e.g., clear digits or letters).

### Expected Run Time
* **GPU:** < 1 second per image.
* **CPU:** ~1-2 seconds per image.
* **Total Demo Time:** Less than 1 minute for a batch of 10 images.

--------------------------------------------------------------------------------
4. INSTRUCTIONS FOR USE (Reproduction)
--------------------------------------------------------------------------------

To reproduce the quantitative results presented in the manuscript, please follow the training instructions below.

### A. Dataset Preparation
Ensure your dataset is organized in the following directory structure:

    dataset/
    ├── input/          # Speckle patterns (Network Input)
    │   ├── 1.bmp
    │   ├── 2.bmp
    │   └── ...
    └── output/         # Ground Truth images (Network Label)
        ├── 1.bmp
        ├── 2.bmp
        └── ...

*(Note: The scripts automatically pair images based on sorted filenames. Ensure `input/1.bmp` corresponds to `output/1.bmp`.)*

### B. How to Run the Software (Training)

**Option 1: Reproducing U-Net Results**
1.  Open `Python_tensorflow/U-Net/main_unet.py`.
2.  Modify the `DATASET_ROOT_DIR` variable in the "USER CONFIGURATION" section to point to your dataset folder.
3.  Run the script:
    ```bash
    python main_unet.py
    ```
4.  **Output:** Trained weights and loss curves will be saved in `./results/experiment_2025`.

**Option 2: Reproducing GAN Results**
1.  Open `Python_tensorflow/GAN/Gan.py`.
2.  Modify `DATASET_INPUT_DIR` and `DATASET_LABEL_DIR` to point to your `input` and `output` folders respectively.
3.  Run the script:
    ```bash
    python Gan.py
    ```
4.  **Output:** Checkpoints and evaluation images will be saved in `./results/gan_experiment`.

### Reproduction Time (Training)
Typical training time for 100 epochs on a dataset of 5,000 images using an NVIDIA RTX 3090:
* **U-Net:** Approximately 1-2 hours.
* **GAN:** Approximately 3-4 hours.
