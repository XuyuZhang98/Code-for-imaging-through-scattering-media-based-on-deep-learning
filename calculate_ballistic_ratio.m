% ==============================================================================
% Ballistic Component Ratio Calculator
% Description: 
%     This script analyzes the optical properties of the scattering medium.
%     It calculates the ratio of ballistic photons (unscattered light) to 
%     total transmitted photons as a function of the phase modulation factor (C).
%     The calculation is based on the Zero-Frequency (DC) component of the 
%     Power Spectrum of the phase mask.
% ==============================================================================

clear; clc; close all;

%% ==============================================================================
%  USER CONFIGURATION
% ==============================================================================
% Path to the Phase Mask data
HEIGHT_MAP_FILE = 'Height3000.mat';

% Physical Parameters
LAMBDA = 0.532e-6;              % Wavelength (meters)

% Analysis Range for Height Factor (C)
% We will sweep C from 1.0 (Strong Scattering) down to 0.01 (Weak Scattering)
C_VALUES = linspace(1.0, 0.01, 100); 

% ==============================================================================

%% 1. Load Data
fprintf('Loading height map from %s...\n', HEIGHT_MAP_FILE);
if exist(HEIGHT_MAP_FILE, 'file')
    load(HEIGHT_MAP_FILE); % This should load a variable named 'Height3000' or 'Height'
else
    error('File not found: %s', HEIGHT_MAP_FILE);
end

% Ensure the variable name is consistent (Handle 'Height3000' or generic 'Height')
if exist('Height3000', 'var')
    HeightData = Height3000;
elseif exist('Height', 'var')
    HeightData = Height;
else
    error('Variable "Height3000" or "Height" not found in the .mat file.');
end

% Convert height to base phase (Physical formula: phi = 2*pi*h / lambda)
Base_Phase = 2 * pi * HeightData / LAMBDA;

% Get dimensions to dynamically find the DC component (Center of the spectrum)
[rows, cols] = size(Base_Phase);
center_r = floor(rows / 2) + 1;
center_c = floor(cols / 2) + 1;

fprintf('Data loaded. Matrix size: %dx%d. Center index: (%d, %d)\n', ...
        rows, cols, center_r, center_c);

%% 2. Calculate Ballistic Ratio Curve
% Initialize array to store weights
Ballistic_Weights = zeros(1, length(C_VALUES));

fprintf('Starting calculation loop...\n');

for i = 1:length(C_VALUES)
    C = C_VALUES(i);
    
    % 1. Apply the Height Factor (C) to generate the Phase Mask
    %    Phasemask = exp(i * C * Phase)
    Phasemask = exp(1j * C * Base_Phase);
    
    % 2. Calculate Power Spectrum (Fourier Transform)
    %    FA_spe: Amplitude Spectrum
    %    FI_spe: Intensity (Power) Spectrum
    FA_spe = abs(fftshift(fft2(ifftshift(Phasemask))));
    FI_spe = FA_spe.^2;
    
    % 3. Normalize Power Spectrum
    %    Normalize by the peak value (usually at DC for weak scattering, 
    %    but here we normalize by the DC component specifically)
    DC_Val = FI_spe(center_r, center_c);
    if DC_Val == 0
        DC_Val = 1e-8; % Avoid division by zero
    end
    FI_spe = FI_spe / DC_Val;
    
    % 4. Estimate Scattered Background at Zero Frequency
    %    Average the power of the 4 nearest neighbors around the center (DC)
    %    to estimate the scattering component overlapping with the ballistic peak.
    %    Neighbors: Left, Right, Up, Down relative to center
    neighbors = [FI_spe(center_r, center_c-6:center_c-1), ...  % Left
                 FI_spe(center_r, center_c+1:center_c+6), ...  % Right
                 FI_spe(center_r-6:center_r-1, center_c)', ... % Up
                 FI_spe(center_r+1:center_r+6, center_c)'];    % Down
             
    FI_0 = mean(neighbors);
    
    % 5. Separation of Ballistic and Scattered Components
    %    Ballistic Power = Total Peak Power - Estimated Scattered Background
    FI_bSum = FI_spe(center_r, center_c) - FI_0;
    
    %    Total Scattered Power = Total Energy - Ballistic Power
    FI_sSum = sum(FI_spe(:)) - FI_bSum;
    
    % 6. Calculate Ballistic Weight (Ratio)
    %    Ratio = Ballistic / (Ballistic + Scattered)
    Ballistic_Weights(i) = FI_bSum / (FI_sSum + FI_bSum);
end

%% 3. Plotting Results
figure('Name', 'Ballistic Ratio vs Height Factor', 'Color', 'w');
plot(C_VALUES, Ballistic_Weights, '-+b', 'LineWidth', 1.5);
grid on;
xlabel('Height Factor (C)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Ballistic Photon Ratio (Weight)', 'FontSize', 12, 'FontWeight', 'bold');
title('Relationship between Scattering Strength and Ballistic Light', 'FontSize', 14);
set(gca, 'XDir', 'reverse'); % Reverse X-axis because smaller C means weaker scattering (more ballistic)

% Print specific values for reference
fprintf('\nReference Values:\n');
ref_indices = [1, 50, 100]; % Start, Middle, End
for idx = ref_indices
    fprintf('  C = %.2f  ->  Ballistic Ratio = %.4f\n', C_VALUES(idx), Ballistic_Weights(idx));
end

fprintf('Calculation finished.\n');