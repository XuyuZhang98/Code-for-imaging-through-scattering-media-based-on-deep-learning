% ==============================================================================
% Simulation of Optical Propagation through Scattering Media
% Description: 
%     This script simulates the free-space optical propagation of objects 
%     (e.g., MNIST/Fashion-MNIST images) through a scattering medium (modeled 
%     by a phase mask) to generate speckle patterns. 
%     It utilizes GPU acceleration for efficient calculation of Fresnel diffraction.
% ==============================================================================

% Clear workspace and close figures
close all; clear; clc;
tic; % Start timer

% Enable CUDA forward compatibility if necessary
parallel.gpu.enableCUDAForwardCompatibility(true);

%% =============================================================================
%  USER CONFIGURATION SECTION
% ==============================================================================

% Directory containing the input Ground Truth images (e.g., label/*.bmp)
% Ensure images are named as '1.bmp', '2.bmp', etc.
INPUT_GT_DIR = 'E:\dataset\fashion_mnist\labels\'; 

% Directory to save the generated Speckle patterns
OUTPUT_SPECKLE_DIR = 'E:\dataset\fashion_mnist\speckle_patterns\'; 

% Path to the Phase Mask data (Scattering Medium)
HEIGHT_MAP_PATH = 'Height3000.mat';

% Simulation Parameters
NUM_SAMPLES = 5000;       % Number of images to process
THRESHOLD = 0.4;          % Threshold for binarization (if needed)

% --- Scattering Control ---
% Factor to scale the height of the medium surface fluctuations.
%   - value = 1.0 : Default scattering strength.
%   - 0 <= value < 1.0 : Weaker scattering (More ballistic components).
HEIGHT_FACTOR = 1.0;      

% ==============================================================================

%% 1. Initialization
% Create output directory if it does not exist
if ~exist(OUTPUT_SPECKLE_DIR, 'dir')
    mkdir(OUTPUT_SPECKLE_DIR);
end

% Initialize GPU
gpuDevice(1); 

%% 2. Define Physical Constants
lambda = 0.532e-6;          % Wavelength (meters)
k = 2 * pi / lambda;        % Wave number
z1 = 0.15;                  % Propagation distance: Source -> Object
z2 = 0.16;                  % Propagation distance: Object -> Medium
z3 = 0.10;                  % Propagation distance: Medium -> CCD

% Spatial Coordinates Parameters
m1 = 1000; n1 = 1e-6;       % Source plane
m2 = 1000; n2 = 13.7e-6;    % Object plane
m22 = 64;                   % Size of the input object image (resized later)
m3 = 1000; n3 = 1e-6;       % Medium plane
m5 = 2048; n5 = 3.45e-6;    % CCD plane (Detector)

% Define Light Source (converted to single precision for GPU speedup)
% 'fov' function defines the field of view or source shape (assumed to be in path)
source = 255 .* fov(m1, m1, m1);
source = gpuArray(single(source)); 

%% 3. Load Scattering Medium (Phase Mask)
if exist(HEIGHT_MAP_PATH, 'file')
    load(HEIGHT_MAP_PATH); % Loads variable 'Height3000'
else
    error('Error: Height map file not found at %s', HEIGHT_MAP_PATH);
end

% Extract the central region and convert to single precision GPU array
Height3000 = gpuArray(single(Height3000)); 
Height2 = Height3000(1001:2000, 1001:2000);

% Pre-calculate the Phase Modulation of the medium
% Formula: exp(1j * k * (Height * factor))
% This controls the ballistic/scattered ratio via HEIGHT_FACTOR.
Medium_Phase = exp(1j * k * (Height2 * HEIGHT_FACTOR));

%% 4. Pre-calculation of Propagation Kernels (Optimization)
% Computing these matrices outside the loop significantly reduces computation time.
% The Fresnel diffraction integral is implemented using matrix multiplication.

% Define coordinate vectors (reshaped as column vectors: m x 1)
vec_m1 = (gpuArray.colon(1, m1) - round(m1/2)).' * n1;
vec_m2 = (gpuArray.colon(1, m2) - round(m2/2)).' * n2;
vec_m3 = (gpuArray.colon(1, m3) - round(m3/2)).' * n3;
vec_m5 = (gpuArray.colon(1, m5) - round(m5/2)).' * n5;

% Calculate Propagation Kernels (A: Column diffraction, B: Row diffraction)
% Kernel Formula: exp(1j * pi / (lambda * z) * (x_out - x_in)^2)

% --- Step 1: Source (m1) to Object (m2) ---
A1 = exp(1j * (pi / (lambda * z1)) * (vec_m2 - vec_m1.').^2); 
B1 = exp(1j * (pi / (lambda * z1)) * (vec_m1 - vec_m2.').^2);

% --- Step 2: Object (m2) to Medium (m3) ---
A2 = exp(1j * (pi / (lambda * z2)) * (vec_m3 - vec_m2.').^2);
B2 = exp(1j * (pi / (lambda * z2)) * (vec_m2 - vec_m3.').^2);

% --- Step 3: Medium (m3) to CCD (m5) ---
A3 = exp(1j * (pi / (lambda * z3)) * (vec_m5 - vec_m3.').^2);
B3 = exp(1j * (pi / (lambda * z3)) * (vec_m3 - vec_m5.').^2);

% Pre-calculate the Illumination Field at the Object plane
% This field is constant since the source does not change
Source_to_Object = A1 * source * B1;

%% 5. Main Simulation Loop
fprintf('Starting simulation for %d samples...\n', NUM_SAMPLES);
fprintf('Scattering Height Factor: %.2f\n', HEIGHT_FACTOR);
pad_w = (1000 - m22) / 2; % Padding width calculation

for I = 1:NUM_SAMPLES
    try
        % --- A. Load and Preprocess Object Image ---
        % Construct file path (Assuming filenames are '1.bmp', '2.bmp', etc.)
        img_filename = fullfile(INPUT_GT_DIR, strcat(num2str(I), '.bmp'));
        
        if ~exist(img_filename, 'file')
            warning('Image file not found: %s. Skipping...', img_filename);
            continue;
        end
        
        % Read image and transfer to GPU
        origin = double(gpuArray(imread(img_filename)));
        
        % Resize object to target dimension (m22 x m22)
        object1 = imresize(origin, [m22, m22]); 
        
        % Pad the object to match the simulation grid size (1000 x 1000)
        object = padarray(object1, [pad_w, pad_w], 0, 'both');
        
        % --- B. Optical Propagation (Physics Core) ---
        
        % 1. Field immediately after the object
        %    Field = Illumination .* Object_Transmittance
        Object_Field = Source_to_Object .* object;
        
        % 2. Propagate from Object to the front surface of the Medium
        Object_to_Medium = A2 * Object_Field * B2;
        
        % 3. Pass through the Scattering Medium (Phase Modulation)
        Medium_Field = Object_to_Medium .* Medium_Phase;
        
        % 4. Propagate from Medium to the CCD plane
        CCD_scat = A3 * Medium_Field * B3;
        
        % 5. Intensity Detection (Square of the amplitude)
        I_CCD_scat = abs(CCD_scat).^2;
        
        % --- C. Save Results ---
        % Normalize the speckle pattern intensity
        max_val = max(I_CCD_scat(:));
        if max_val > 0
            I_CCD_normalized = I_CCD_scat / max_val;
        else
            I_CCD_normalized = I_CCD_scat;
        end
        
        % Write to file
        save_filename = fullfile(OUTPUT_SPECKLE_DIR, strcat(num2str(I), '.bmp'));
        imwrite(gather(I_CCD_normalized), save_filename);
        
        % Display progress
        if mod(I, 100) == 0
            fprintf('Processed: %d / %d\n', I, NUM_SAMPLES);
        end
        
    catch ME
        fprintf('Error processing sample %d: %s\n', I, ME.message);
    end
end

% Calculate total elapsed time
elapsed_time = toc;
fprintf('Simulation completed in %.2f seconds.\n', elapsed_time);