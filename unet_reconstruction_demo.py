"""
GUI Inference Tool for Scattering Imaging Reconstruction
Author: [Xuyu Zhang]
Description:
    This script provides a Graphical User Interface (GUI) based on Tkinter to facilitate
    interactive model inference. Users can load trained weights, batch select speckle
    patterns, and generate reconstructed images in either grayscale or binary formats.
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, \
    BatchNormalization, Activation
from tensorflow.keras.models import Model

# --- GPU Configuration ---
# Allow dynamic GPU memory allocation to prevent OOM errors
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# ==========================================
# 1. Network Architecture (U-Net)
# ==========================================
class UNetModelBuilder:
    def __init__(self, height=256, width=256, channels=1):
        """
        Initialize U-Net parameters.
        """
        self.height = height
        self.width = width
        self.channels = channels
        self.shape = (self.height, self.width, self.channels)

    def build_unet(self, n_filters=16, dropout=0.1, batchnorm=True, padding='same'):
        """
        Constructs the U-Net model architecture.
        """

        # Helper function for Convolutional Block
        def conv2d_block(input_tensor, n_filters=16, kernel_size=3, batchnorm=True, padding='same'):
            # First Convolution
            x = Conv2D(n_filters, kernel_size, padding=padding)(input_tensor)
            if batchnorm: x = BatchNormalization()(x)
            x = Activation('relu')(x)

            # Second Convolution
            x = Conv2D(n_filters, kernel_size, padding=padding)(x)
            if batchnorm: x = BatchNormalization()(x)
            X = Activation('relu')(x)
            return X

        img = Input(shape=self.shape)

        # --- Contracting Path (Encoder) ---
        c1 = conv2d_block(img, n_filters=n_filters * 1, batchnorm=batchnorm)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(dropout * 0.5)(p1)

        c2 = conv2d_block(p1, n_filters=n_filters * 2, batchnorm=batchnorm)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout)(p2)

        c3 = conv2d_block(p2, n_filters=n_filters * 4, batchnorm=batchnorm)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(dropout)(p3)

        c4 = conv2d_block(p3, n_filters=n_filters * 8, batchnorm=batchnorm)
        p4 = MaxPooling2D((2, 2))(c4)
        p4 = Dropout(dropout)(p4)

        # Bottleneck
        c5 = conv2d_block(p4, n_filters=n_filters * 16, batchnorm=batchnorm)

        # --- Expansive Path (Decoder) ---
        u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = conv2d_block(u6, n_filters=n_filters * 8, batchnorm=batchnorm)

        u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = conv2d_block(u7, n_filters=n_filters * 4, batchnorm=batchnorm)

        u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = conv2d_block(u8, n_filters=n_filters * 2, batchnorm=batchnorm)

        u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1])
        u9 = Dropout(dropout)(u9)
        c9 = conv2d_block(u9, n_filters=n_filters * 1, batchnorm=batchnorm)

        # Output Layer (Sigmoid for [0, 1] range)
        output = Conv2D(1, (1, 1), activation='sigmoid')(c9)
        return Model(img, output)


# ==========================================
# 2. GUI Interface Class
# ==========================================
class ScatteringInferenceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Deep Learning Scattering Imaging Inference Tool")
        self.root.geometry("720x620")

        # Variable Storage
        self.weight_path = tk.StringVar()
        self.input_files = []
        self.save_dir = tk.StringVar()
        self.is_saving = tk.BooleanVar(value=True)

        # Save Mode: 'gray' (Continuous) or 'binary' (Thresholded)
        self.save_mode = tk.StringVar(value="gray")
        self.bin_threshold = tk.DoubleVar(value=0.5)

        self.model = None
        self.builder = UNetModelBuilder()

        # Style Configuration (Using Arial for compatibility)
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 9))
        style.configure("TLabel", font=("Arial", 9))

        self.create_widgets()

    def create_widgets(self):
        """Builds the GUI components."""

        # --- Step 1: Model Configuration ---
        frame_top = ttk.LabelFrame(self.root, text="Step 1: Model Configuration", padding=10)
        frame_top.pack(fill="x", padx=10, pady=5)

        ttk.Label(frame_top, text="Weights File (.keras):").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame_top, textvariable=self.weight_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(frame_top, text="Browse...", command=self.select_weights).grid(row=0, column=2)

        # --- Step 2: Data Selection ---
        frame_mid = ttk.LabelFrame(self.root, text="Step 2: Input Speckle Selection", padding=10)
        frame_mid.pack(fill="both", expand=True, padx=10, pady=5)

        btn_frame = ttk.Frame(frame_mid)
        btn_frame.pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Add Images", command=self.add_images).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Clear List", command=self.clear_images).pack(side="left", padx=5)
        self.lbl_count = ttk.Label(btn_frame, text="Selected: 0")
        self.lbl_count.pack(side="right", padx=10)

        self.listbox = tk.Listbox(frame_mid, height=8, selectmode=tk.EXTENDED)
        self.listbox.pack(fill="both", expand=True, pady=5)

        # --- Step 3: Prediction & Save Settings ---
        frame_bottom = ttk.LabelFrame(self.root, text="Step 3: Prediction & Save Settings", padding=10)
        frame_bottom.pack(fill="x", padx=10, pady=5)

        # Row 1: Path Selection
        f3_r1 = ttk.Frame(frame_bottom)
        f3_r1.pack(fill="x", pady=2)
        chk_save = ttk.Checkbutton(f3_r1, text="Enable Saving", variable=self.is_saving,
                                   command=self.toggle_save_controls)
        chk_save.pack(side="left")
        self.entry_save = ttk.Entry(f3_r1, textvariable=self.save_dir, width=40)
        self.entry_save.pack(side="left", padx=5)
        self.btn_save_browse = ttk.Button(f3_r1, text="Select Folder...", command=self.select_save_dir)
        self.btn_save_browse.pack(side="left")

        # Row 2: Output Format
        f3_r2 = ttk.Frame(frame_bottom)
        f3_r2.pack(fill="x", pady=5)
        ttk.Label(f3_r2, text="Output Format: ").pack(side="left")

        # Radio Buttons
        rb_gray = ttk.Radiobutton(f3_r2, text="Grayscale (Continuous)", variable=self.save_mode, value="gray",
                                  command=self.toggle_threshold)
        rb_gray.pack(side="left", padx=10)

        rb_bin = ttk.Radiobutton(f3_r2, text="Binary (Threshold)", variable=self.save_mode, value="binary",
                                 command=self.toggle_threshold)
        rb_bin.pack(side="left", padx=10)

        # Threshold Input
        ttk.Label(f3_r2, text="Threshold:").pack(side="left", padx=(10, 2))
        self.entry_thresh = ttk.Entry(f3_r2, textvariable=self.bin_threshold, width=6)
        self.entry_thresh.pack(side="left")
        self.toggle_threshold()  # Initialize state

        # Predict Button
        self.btn_predict = ttk.Button(frame_bottom, text="Start Reconstruction", command=self.start_prediction_thread)
        self.btn_predict.pack(fill="x", pady=10)

        # --- Logging Area ---
        self.log_text = scrolledtext.ScrolledText(self.root, height=8, state='disabled', font=("Consolas", 9))
        self.log_text.pack(fill="x", padx=10, pady=5)

    def log(self, msg):
        """Updates the log window in a thread-safe manner."""
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def select_weights(self):
        filename = filedialog.askopenfilename(title="Select Model Weights",
                                              filetypes=[("Model Weights", "*.keras *.h5")])
        if filename:
            self.weight_path.set(filename)
            self.log(f"Loaded weights: {os.path.basename(filename)}")

    def add_images(self):
        filenames = filedialog.askopenfilenames(title="Select Speckle Images",
                                                filetypes=[("Images", "*.bmp *.png *.jpg *.tif")])
        if filenames:
            for f in filenames:
                if f not in self.input_files:
                    self.input_files.append(f)
                    self.listbox.insert(tk.END, os.path.basename(f))
            self.lbl_count.config(text=f"Selected: {len(self.input_files)}")

    def clear_images(self):
        self.input_files = []
        self.listbox.delete(0, tk.END)
        self.lbl_count.config(text="Selected: 0")

    def toggle_save_controls(self):
        state = 'normal' if self.is_saving.get() else 'disabled'
        self.entry_save.config(state=state)
        self.btn_save_browse.config(state=state)

    def toggle_threshold(self):
        """Enable/Disable threshold entry based on selected mode."""
        if self.save_mode.get() == 'binary':
            self.entry_thresh.config(state='normal')
        else:
            self.entry_thresh.config(state='disabled')

    def select_save_dir(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory: self.save_dir.set(directory)

    def start_prediction_thread(self):
        """Runs prediction in a separate thread to prevent GUI freezing."""
        threading.Thread(target=self.run_prediction, daemon=True).start()

    def run_prediction(self):
        weight_file = self.weight_path.get()
        if not weight_file or not os.path.exists(weight_file):
            messagebox.showerror("Error", "Invalid weights file!")
            return
        if not self.input_files:
            messagebox.showerror("Error", "No images selected!")
            return

        save_path = self.save_dir.get()
        if self.is_saving.get() and not save_path:
            messagebox.showerror("Error", "Please select a save directory!")
            return

        self.btn_predict.config(state='disabled')

        try:
            # 1. Initialize Model and Load Weights
            if self.model is None:
                self.log("Initializing model architecture...")
                self.model = self.builder.build_unet()
                self.log("Loading weights...")
                self.model.load_weights(weight_file)

            # Get current settings
            mode = self.save_mode.get()
            try:
                thresh = float(self.bin_threshold.get())
            except:
                thresh = 0.5

            self.log(
                f"Starting inference... Mode: {'Binary (Thresh=' + str(thresh) + ')' if mode == 'binary' else 'Grayscale'}")

            # 2. Batch Processing
            for idx, img_path in enumerate(self.input_files):
                filename = os.path.basename(img_path)

                # Preprocessing: Read -> Grayscale -> Resize -> Normalize
                img = Image.open(img_path).convert('L')
                img = img.resize((256, 256))
                img_arr = np.array(img, dtype='float32') / 255.0
                # Expand dims for model input: (1, 256, 256, 1)
                img_input = np.expand_dims(np.expand_dims(img_arr, axis=0), axis=-1)

                # Inference
                pred = self.model.predict(img_input, verbose=0)
                pred = np.squeeze(pred)  # Remove batch dim: (256, 256)

                # Post-processing based on mode
                if mode == 'binary':
                    # Binary mode: 0 or 255 based on threshold
                    result_arr = (pred > thresh).astype(np.uint8) * 255
                else:
                    # Grayscale mode: Scale 0-1 to 0-255
                    result_arr = np.clip(pred * 255.0, 0, 255).astype(np.uint8)

                # Save Results
                if self.is_saving.get():
                    save_name = os.path.join(save_path, f"{filename}")
                    base, ext = os.path.splitext(save_name)
                    # Force valid extension for saving
                    if ext.lower() not in ['.png', '.jpg', '.bmp']:
                        save_name = base + ".png"

                    Image.fromarray(result_arr).save(save_name)
                    self.log(f"[{idx + 1}/{len(self.input_files)}] Saved: {os.path.basename(save_name)}")

            messagebox.showinfo("Success", "Inference completed successfully!")
            self.log("--- Task Finished ---")

        except Exception as e:
            self.log(f"Error: {e}")
            messagebox.showerror("Execution Error", str(e))
        finally:
            self.btn_predict.config(state='normal')


if __name__ == '__main__':
    root = tk.Tk()
    app = ScatteringInferenceGUI(root)
    root.mainloop()