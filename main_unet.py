"""
Author: [Xuyu Zhang]
Description: 
    This script implements a U-Net architecture for reconstructing objects from speckle patterns.
    It includes data loading, model construction, transfer learning options, and the training loop.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from glob import glob
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2DTranspose
from tensorflow.keras.layers import Input, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Allow dynamic GPU memory allocation
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# ==============================================================================
# USER CONFIGURATION (Please modify paths before running)
# ==============================================================================
# Directory containing the dataset (input/output folders)
DATASET_ROOT_DIR = r"./dataset/speckle_data"

# Directory to save training results
RESULT_SAVE_DIR = r"./results/experiment_2025"

# Path to pretrained model for Transfer Learning (Optional)
# Set to None if training from scratch. 
# Example: r"./previous_works/best_model.keras"
PRETRAINED_MODEL_PATH = None

# Training Hyperparameters
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 1
LEARNING_RATE = 0.002
BATCH_SIZE = 16
EPOCHS = 100


# ==============================================================================


class U_Net():
    def __init__(self):
        """
        Initialize the U-Net model parameters, optimizer, and compilation.
        """
        # Set basic image parameters
        self.height = IMG_HEIGHT
        self.width = IMG_WIDTH
        self.channels = IMG_CHANNELS
        self.shape = (self.height, self.width, self.channels)

        # Initialize the optimizer
        self.optimizer = Adam(learning_rate=LEARNING_RATE, beta_1=0.5)

        # Build and compile the U-Net model
        self.unet = self.build_unet()
        self.compile_model()
        self.unet.summary()

    def compile_model(self):
        """
        Compiles the model. Separated into a function to allow re-compilation 
        after freezing layers during transfer learning.
        """
        self.unet.compile(loss='mse',
                          optimizer=self.optimizer,
                          metrics=[self.metric_fun])

    def build_unet(self, n_filters=16, dropout=0.1, batchnorm=True, padding='same'):
        """
        Construct the U-Net architecture.
        """

        # Define a reusable convolutional block
        def conv2d_block(input_tensor, n_filters=16, kernel_size=3, batchnorm=True, padding='same'):
            # The first convolutional layer
            x = Conv2D(n_filters, kernel_size, padding=padding)(input_tensor)
            if batchnorm:
                x = BatchNormalization()(x)
            x = Activation('relu')(x)

            # The second convolutional layer
            x = Conv2D(n_filters, kernel_size, padding=padding)(x)
            if batchnorm:
                x = BatchNormalization()(x)
            X = Activation('relu')(x)
            return X

        # Define input layer
        img = Input(shape=self.shape)

        # --- Contracting path (Encoder) ---
        c1 = conv2d_block(img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm, padding=padding)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(dropout * 0.5)(p1)

        c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm, padding=padding)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout)(p2)

        c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm, padding=padding)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(dropout)(p3)

        c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm, padding=padding)
        p4 = MaxPooling2D((2, 2))(c4)
        p4 = Dropout(dropout)(p4)

        c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm, padding=padding)

        # --- Extending path (Decoder) ---
        u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm, padding=padding)

        u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm, padding=padding)

        u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c8)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm, padding=padding)

        u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1])
        u9 = Dropout(dropout)(u9)
        c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm, padding=padding)

        # Output layer
        output = Conv2D(1, (1, 1), activation='sigmoid')(c9)

        return Model(img, output)

    def metric_fun(self, y_true, y_pred):
        """
        Custom metric function (similar to Dice coefficient) for monitoring training.
        """
        threshold = 0.1
        fz = tf.reduce_sum(2 * y_true * tf.cast(tf.greater(y_pred, threshold), tf.float32)) + 1e-8
        fm = tf.reduce_sum(y_true + tf.cast(tf.greater(y_pred, threshold), tf.float32)) + 1e-8
        return fz / fm

    def load_data(self):
        """
        Load and preprocess the dataset.
        Returns:
            x_train, x_label: Training data and labels
            y_train, y_label: Validation data and labels
        """
        x_train = []  # List to store input images (Speckle)
        x_label = []  # List to store label images (Ground Truth)

        # Define paths using the global configuration
        speckle_path = os.path.join(DATASET_ROOT_DIR, "input", "*.bmp")
        gt_path = os.path.join(DATASET_ROOT_DIR, "output", "*.bmp")

        # Load input images (Speckle patterns)
        for file in sorted(glob(speckle_path)):
            img = np.array(Image.open(file), dtype='float32') / 255.0
            x_train.append(img)

        # Load label images (Ground Truth)
        for file in sorted(glob(gt_path)):
            img = np.array(Image.open(file), dtype='float32') / 255.0
            x_label.append(img)

        # Expand dimensions
        x_train = np.expand_dims(np.array(x_train), axis=3)
        x_label = np.expand_dims(np.array(x_label), axis=3)

        # Shuffle the dataset
        seed = 116
        np.random.seed(seed)
        np.random.shuffle(x_train)
        np.random.seed(seed)
        np.random.shuffle(x_label)

        # Split dataset: 90% for training, 10% for validation
        total_samples = x_train.shape[0]
        split_idx = int(total_samples * 0.9)

        return x_train[:split_idx, :], x_label[:split_idx, :], x_train[split_idx:, :], x_label[split_idx:, :]

    def train(self, epochs=EPOCHS, batch_size=BATCH_SIZE):
        """
        Execute the training process and save the results.
        """
        # Create directory for saving results
        os.makedirs(RESULT_SAVE_DIR, exist_ok=True)

        # --- Transfer Learning Section ---
        # Load pre-trained weights and freeze layers if path is provided
        if PRETRAINED_MODEL_PATH is not None and os.path.exists(PRETRAINED_MODEL_PATH):
            print(f"Loading pretrained weights from: {PRETRAINED_MODEL_PATH}")
            self.unet.load_weights(PRETRAINED_MODEL_PATH)

            # Freeze all layers in the existing model
            for layer in self.unet.layers:
                layer.trainable = False

            # IMPORTANT: Re-compile the model for 'trainable=False' to take effect
            self.compile_model()
            print("Pretrained layers frozen and model re-compiled.")
        # ---------------------------------

        # Load data
        x_train, x_label, y_train, y_label = self.load_data()

        # Define callbacks
        checkpoint_path = os.path.join(RESULT_SAVE_DIR, 'best_model.keras')
        callbacks = [
            EarlyStopping(patience=10, verbose=2),
            ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.000005, verbose=2),
            ModelCheckpoint(checkpoint_path, verbose=2, save_weights_only=False, save_best_only=True)
        ]

        # Train the model
        results = self.unet.fit(x_train, x_label,
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=2,
                                callbacks=callbacks,
                                validation_split=0.1,
                                shuffle=True)

        # Plot and save loss/metric curves
        loss = results.history['loss']
        val_loss = results.history['val_loss']
        metric = results.history['metric_fun']
        val_metric = results.history['val_metric_fun']

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        x = np.arange(len(loss))

        # Loss curve
        plt.subplot(121)
        plt.plot(x, loss, label='Train Loss')
        plt.plot(x, val_loss, label='Val Loss')
        plt.title("Loss Curve")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")

        # Metric curve
        plt.subplot(122)
        plt.plot(x, metric, label='Train Metric')
        plt.plot(x, val_metric, label='Val Metric')
        plt.title("Metric Curve")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Dice Metric")

        curve_save_path = os.path.join(RESULT_SAVE_DIR, 'learning_curve.png')
        fig.savefig(curve_save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"Training finished. Results saved to {RESULT_SAVE_DIR}")


if __name__ == '__main__':
    unet = U_Net()
    unet.train()