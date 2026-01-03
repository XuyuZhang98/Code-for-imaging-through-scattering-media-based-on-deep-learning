# MATLAB Simulation & Analysis Package for Scattering Imaging


--------------------------------------------------------------------------------
1. OVERVIEW
--------------------------------------------------------------------------------
This package contains the MATLAB source codes for simulating the physical process 
of optical propagation through scattering media and characterizing the medium's 
properties.

It consists of two main scripts:

1. speckle_simulation_propagation.m: 
   The core simulation engine. It models the multi-stage Fresnel diffraction 
   process to generate paired datasets (Ground Truth <-> Speckle Patterns) 
   used for training the deep learning model.

2. calculate_ballistic_ratio.m: 
   A characterization tool. It analyzes the phase mask data to quantify the 
   relationship between the scattering medium's height fluctuations and the 
   ratio of ballistic (unscattered) photons.

--------------------------------------------------------------------------------
2. SYSTEM REQUIREMENTS
--------------------------------------------------------------------------------

### Software
- MATLAB (R2023a or later recommended).
- **Parallel Computing Toolbox** (MANDATORY).
  *The simulation script relies on `gpuArray` and CUDA-enabled functions.*

### Hardware
- **GPU**: A compatible NVIDIA GPU with CUDA support is required.
- **VRAM**: Minimum 4GB GPU Memory is recommended for the default simulation 
  grid size (2048 x 2048).

--------------------------------------------------------------------------------
3. FILE PREPARATION
--------------------------------------------------------------------------------

### A. Phase Mask Data (Essential)
The simulation requires a pre-defined phase mask file to model the scattering 
surface.
* Ensure the file `Height3000.mat` is placed in the root directory of these 
  scripts.

### B. Input Source Images
Prepare your clean object images (Ground Truth) for simulation (e.g., MNIST or  bitmaps). The directory structure should look like this:

    dataset/
    ├── labels/              # Source images (e.g., 1.bmp, 2.bmp...)
    └── speckle_patterns/    # (Empty folder) Destination for generated outputs

--------------------------------------------------------------------------------
4. USAGE INSTRUCTIONS
--------------------------------------------------------------------------------

### Part A: Generating the Dataset (speckle_simulation_propagation.m)

1. Open `speckle_simulation_propagation.m`.
2. Locate the "USER CONFIGURATION SECTION".
3. Modify the paths to match your local environment:
   
   INPUT_GT_DIR = 'E:\path\to\your\labels\'; 
   OUTPUT_SPECKLE_DIR = 'E:\path\to\your\speckle_patterns\'; 
   HEIGHT_MAP_PATH = 'Height3000.mat';

4. Configure the Scattering Strength (`HEIGHT_FACTOR`):
   
   HEIGHT_FACTOR = 1.0; 
   
   * Note: This factor scales the height fluctuations of the medium.
     - 1.0 = Default strong scattering.
     - 0 ~ 1.0 = Weaker scattering (More ballistic components).
     - You can use Part B (below) to determine the exact ballistic ratio for 
       a specific factor.

5. Run the script. The progress will be displayed in the Command Window.

---

### Part B: Characterizing the Medium (calculate_ballistic_ratio.m)

Use this script to understand the physical properties corresponding to different 
`HEIGHT_FACTOR` values.

1. Open `calculate_ballistic_ratio.m`.
2. Ensure `HEIGHT_MAP_FILE` points to the correct .mat file.
3. Run the script.
4. Output:
   - A plot window: "Ballistic Photon Ratio vs. Height Factor".
   - Console output: Reference values listing the ballistic ratio for specific 
     factors (C values).

--------------------------------------------------------------------------------
5. OUTPUT DATA
--------------------------------------------------------------------------------
- **Simulation**: Generates .bmp speckle patterns in `OUTPUT_SPECKLE_DIR`. 
  These should be used as the "Input" for the Neural Network, while the images 
  in `INPUT_GT_DIR` serve as the "Label/Output".

- **Analysis**: Generates a performance curve allowing the user to select an 
  appropriate scattering difficulty level for the experiment.