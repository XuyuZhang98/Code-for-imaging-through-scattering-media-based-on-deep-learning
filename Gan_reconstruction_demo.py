"""
GUI Inference Tool for GAN-based Speckle Reconstruction (Final Fix)
Framework: TensorFlow / Keras
Author: [Anonymous for Review]
Date: 2025-12-30
Description:
    This script provides a Graphical User Interface (GUI) to load the trained
    GAN weights. It includes a specific fix for loading HDF5 weights into
    subclassed Models by forcing a model build before loading.
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras import layers, models, Model, Input
from tensorflow.keras.initializers import RandomNormal

# --- GPU Configuration ---
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# ==============================================================================
# 1. MODEL ARCHITECTURES (Exact Match)
# ==============================================================================

def build_unet_generator(input_shape=(256, 256, 1)):
    """
    Defines the 5-level U-Net Generator used in training.
    """
    inputs = Input(shape=input_shape)

    def conv_block(x, filters, apply_bn=True):
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        if apply_bn: x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        if apply_bn: x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    # --- Encoder ---
    c1 = conv_block(inputs, 16)
    p1 = layers.MaxPooling2D()(c1)
    p1 = layers.Dropout(0.1)(p1)

    c2 = conv_block(p1, 32)
    p2 = layers.MaxPooling2D()(c2)
    p2 = layers.Dropout(0.1)(p2)

    c3 = conv_block(p2, 64)
    p3 = layers.MaxPooling2D()(c3)
    p3 = layers.Dropout(0.1)(p3)

    c4 = conv_block(p3, 128)
    p4 = layers.MaxPooling2D()(c4)
    p4 = layers.Dropout(0.1)(p4)

    c5 = conv_block(p4, 256)
    p5 = layers.MaxPooling2D()(c5)
    p5 = layers.Dropout(0.1)(p5)

    # --- Bottleneck ---
    c6 = conv_block(p5, 512)

    # --- Decoder ---
    u7 = layers.Conv2DTranspose(256, 3, strides=2, padding='same')(c6)
    u7 = layers.Concatenate()([u7, c5])
    c7 = conv_block(u7, 256)

    u8 = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(c7)
    u8 = layers.Concatenate()([u8, c4])
    c8 = conv_block(u8, 128)

    u9 = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(c8)
    u9 = layers.Concatenate()([u9, c3])
    c9 = conv_block(u9, 64)

    u10 = layers.Conv2DTranspose(32, 3, strides=2, padding='same')(c9)
    u10 = layers.Concatenate()([u10, c2])
    c10 = conv_block(u10, 32)

    u11 = layers.Conv2DTranspose(16, 3, strides=2, padding='same')(c10)
    u11 = layers.Concatenate()([u11, c1])
    c11 = conv_block(u11, 16)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c11)
    return Model(inputs=inputs, outputs=outputs, name="Generator")

def build_discriminator(input_shape=(256, 256, 1)):
    """
    Discriminator architecture needed to reconstruct the GAN wrapper.
    """
    img_input = Input(shape=input_shape, name='target_img')
    cond_input = Input(shape=input_shape, name='condition_img')
    combined = layers.Concatenate()([img_input, cond_input])

    x = layers.Conv2D(32, 4, strides=2, padding='same', kernel_initializer=RandomNormal(0, 0.02))(combined)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(64, 4, strides=2, padding='same', kernel_initializer=RandomNormal(0, 0.02))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(128, 4, strides=2, padding='same', kernel_initializer=RandomNormal(0, 0.02))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(256, 4, strides=2, padding='same', kernel_initializer=RandomNormal(0, 0.02))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    return Model(inputs=[img_input, cond_input], outputs=x, name="Discriminator")

class SpeckleGAN(Model):
    """
    Wrapper class required to load weights saved by gan.fit().
    """
    def __init__(self, generator, discriminator):
        super(SpeckleGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    # FIX: Add a call method to allow the model to be built
    def call(self, inputs):
        return self.generator(inputs)

# ==============================================================================
# 2. GUI CLASS
# ==============================================================================
class GANInferenceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GAN-based Speckle Reconstruction Tool")
        self.root.geometry("720x620")

        # Variables
        self.weight_path = tk.StringVar()
        self.input_files = []
        self.save_dir = tk.StringVar()
        self.is_saving = tk.BooleanVar(value=True)
        self.save_mode = tk.StringVar(value="gray")
        self.bin_threshold = tk.DoubleVar(value=0.5)

        self.model = None

        # Style
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 9))
        style.configure("TLabel", font=("Arial", 9))

        self.create_widgets()

    def create_widgets(self):
        # --- Step 1: Model Configuration ---
        frame_top = ttk.LabelFrame(self.root, text="Step 1: Load GAN Checkpoint", padding=10)
        frame_top.pack(fill="x", padx=10, pady=5)
        ttk.Label(frame_top, text="Weights File (.h5):").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame_top, textvariable=self.weight_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(frame_top, text="Browse...", command=self.select_weights).grid(row=0, column=2)

        # --- Step 2: Data Selection ---
        frame_mid = ttk.LabelFrame(self.root, text="Step 2: Select Speckle Images", padding=10)
        frame_mid.pack(fill="both", expand=True, padx=10, pady=5)
        btn_frame = ttk.Frame(frame_mid)
        btn_frame.pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Add Images", command=self.add_images).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Clear List", command=self.clear_images).pack(side="left", padx=5)
        self.lbl_count = ttk.Label(btn_frame, text="Selected: 0")
        self.lbl_count.pack(side="right", padx=10)
        self.listbox = tk.Listbox(frame_mid, height=8, selectmode=tk.EXTENDED)
        self.listbox.pack(fill="both", expand=True, pady=5)

        # --- Step 3: Prediction Settings ---
        frame_bottom = ttk.LabelFrame(self.root, text="Step 3: Prediction & Output", padding=10)
        frame_bottom.pack(fill="x", padx=10, pady=5)

        f3_r1 = ttk.Frame(frame_bottom)
        f3_r1.pack(fill="x", pady=2)
        chk_save = ttk.Checkbutton(f3_r1, text="Enable Saving", variable=self.is_saving, command=self.toggle_save_controls)
        chk_save.pack(side="left")
        self.entry_save = ttk.Entry(f3_r1, textvariable=self.save_dir, width=40)
        self.entry_save.pack(side="left", padx=5)
        self.btn_save_browse = ttk.Button(f3_r1, text="Select Folder...", command=self.select_save_dir)
        self.btn_save_browse.pack(side="left")

        f3_r2 = ttk.Frame(frame_bottom)
        f3_r2.pack(fill="x", pady=5)
        ttk.Label(f3_r2, text="Output Format: ").pack(side="left")
        rb_gray = ttk.Radiobutton(f3_r2, text="Grayscale", variable=self.save_mode, value="gray", command=self.toggle_threshold)
        rb_gray.pack(side="left", padx=10)
        rb_bin = ttk.Radiobutton(f3_r2, text="Binary", variable=self.save_mode, value="binary", command=self.toggle_threshold)
        rb_bin.pack(side="left", padx=10)
        ttk.Label(f3_r2, text="Threshold:").pack(side="left", padx=(10, 2))
        self.entry_thresh = ttk.Entry(f3_r2, textvariable=self.bin_threshold, width=6)
        self.entry_thresh.pack(side="left")
        self.toggle_threshold()

        self.btn_predict = ttk.Button(frame_bottom, text="Run Reconstruction", command=self.start_prediction_thread)
        self.btn_predict.pack(fill="x", pady=10)

        self.log_text = scrolledtext.ScrolledText(self.root, height=8, state='disabled', font=("Consolas", 9))
        self.log_text.pack(fill="x", padx=10, pady=5)

    def log(self, msg):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def select_weights(self):
        filename = filedialog.askopenfilename(title="Select GAN Checkpoint", filetypes=[("Weights", "*.h5 *.keras")])
        if filename:
            self.weight_path.set(filename)
            self.log(f"Selected: {os.path.basename(filename)}")

    def add_images(self):
        filenames = filedialog.askopenfilenames(title="Select Images", filetypes=[("Images", "*.bmp *.png *.jpg *.tif")])
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
        if self.save_mode.get() == 'binary':
            self.entry_thresh.config(state='normal')
        else:
            self.entry_thresh.config(state='disabled')

    def select_save_dir(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory: self.save_dir.set(directory)

    def start_prediction_thread(self):
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
            # 1. Initialize Model Strategy
            if self.model is None:
                self.log("Initializing GAN Structure...")

                # Reconstruct models
                gen = build_unet_generator()
                disc = build_discriminator()
                gan_wrapper = SpeckleGAN(gen, disc)

                self.log("Building model computation graph...")
                # CRITICAL FIX: Run a dummy forward pass to "build" the model layers
                dummy_in = tf.zeros((1, 256, 256, 1))
                # Call the wrapper (which calls generator)
                _ = gan_wrapper(dummy_in)

                self.log("Loading weights...")
                # Now load weights (TensorFlow now knows the variables exist)
                gan_wrapper.load_weights(weight_file)
                self.log("Weights loaded successfully.")

                # Extract Generator
                self.model = gan_wrapper.generator

            # Settings
            mode = self.save_mode.get()
            try:
                thresh = float(self.bin_threshold.get())
            except:
                thresh = 0.5
            self.log(f"Starting inference... Mode: {mode}")

            # 2. Processing
            for idx, img_path in enumerate(self.input_files):
                filename = os.path.basename(img_path)

                # Preprocess: Load via PIL -> Grayscale -> Resize -> Normalize
                img = Image.open(img_path).convert('L')
                img = img.resize((256, 256))
                img_arr = np.array(img, dtype='float32') / 255.0
                img_input = np.expand_dims(np.expand_dims(img_arr, axis=0), axis=-1)

                # Inference
                pred = self.model.predict(img_input, verbose=0)
                pred = np.squeeze(pred)

                # Post-process
                if mode == 'binary':
                    result_arr = (pred > thresh).astype(np.uint8) * 255
                else:
                    result_arr = np.clip(pred * 255.0, 0, 255).astype(np.uint8)

                # Save
                if self.is_saving.get():
                    save_name = os.path.join(save_path, f"{filename}")
                    base, ext = os.path.splitext(save_name)
                    if ext.lower() not in ['.png', '.jpg', '.bmp']:
                        save_name = base + ".png"

                    Image.fromarray(result_arr).save(save_name)
                    self.log(f"[{idx + 1}/{len(self.input_files)}] Saved: {os.path.basename(save_name)}")

            messagebox.showinfo("Success", "All images processed!")
            self.log("--- Task Finished ---")

        except Exception as e:
            self.log(f"Error: {e}")
            print(e)
            messagebox.showerror("Execution Error", f"Check console for details.\n{str(e)}")
        finally:
            self.btn_predict.config(state='normal')


if __name__ == '__main__':
    root = tk.Tk()
    app = GANInferenceGUI(root)
    root.mainloop()