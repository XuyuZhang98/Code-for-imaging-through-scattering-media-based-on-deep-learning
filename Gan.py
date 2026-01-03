"""
Framework: TensorFlow / Keras
Description: 
    This script implements a Conditional Generative Adversarial Network (cGAN)
    for reconstructing object images from speckle patterns.

    Key Features:
    1. Generator: 5-layer U-Net architecture with skip connections.
    2. Discriminator: Deep Convolutional PatchGAN-like classifier.
    3. Loss Function: Combination of Adversarial Loss and Pearson Correlation Coefficient (PCC) Loss.
    4. Transfer Learning: Supports loading pretrained weights and freezing specific layers.
"""

import os
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal

# ==============================================================================
# USER CONFIGURATION
# ==============================================================================

# GPU Memory Management
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Dataset Paths
# Note: 'input' folder contains Speckle images; 'output' folder contains Ground Truth images.
DATASET_INPUT_DIR = r"D:/Speckle_Data/train/input"   # Network Input (Speckle)
DATASET_LABEL_DIR = r"D:/Speckle_Data/train/output"  # Network Label (Ground Truth)

# Output Directories
RESULT_DIR = r"./results/gan_experiment"
CHECKPOINT_DIR = os.path.join(RESULT_DIR, "checkpoints")
LOG_DIR = os.path.join(RESULT_DIR, "logs")
EVAL_DIR = os.path.join(RESULT_DIR, "evaluation")

# Transfer Learning Configuration
# Path to pretrained generator weights (e.g., r"./weights/best_generator.h5").
# Set to None to train from scratch.
PRETRAINED_GENERATOR_PATH = None

# Hyperparameters
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 16       # Adjust based on GPU memory
EPOCHS = 100
LR_GENERATOR = 1e-4
LR_DISCRIMINATOR = 1e-5

# Loss Function Weights
# Total Generator Loss = (LAMBDA_ADV * Adversarial_Loss) + (LAMBDA_PCC * PCC_Loss)
LAMBDA_ADV = 0.05
LAMBDA_PCC = 1.0

# ==============================================================================
# LOSS FUNCTIONS
# ==============================================================================

def pcc_loss(y_true, y_pred):
    """
    Pearson Correlation Coefficient (PCC) Loss.
    Objective: Maximize the structural correlation between prediction and ground truth.
    Formula: Loss = 1 - PCC
    """
    # Flatten images to 1D vectors
    x = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])

    # Calculate means
    mx = tf.reduce_mean(x, axis=1, keepdims=True)
    my = tf.reduce_mean(y, axis=1, keepdims=True)

    # Center variables
    xm = x - mx
    ym = y - my

    # Calculate correlation
    numerator = tf.reduce_sum(xm * ym, axis=1)
    denominator = tf.sqrt(tf.reduce_sum(tf.square(xm), axis=1)) * \
                  tf.sqrt(tf.reduce_sum(tf.square(ym), axis=1))

    # Add epsilon to prevent division by zero
    pcc = numerator / (denominator + 1e-8)

    # Minimize (1 - PCC)
    return tf.reduce_mean(1.0 - pcc)

def generator_loss(fake_output, real_img, fake_img):
    """
    Combined Generator Loss.
    """
    # 1. Adversarial Loss (Binary Cross Entropy)
    # The generator tries to make the discriminator classify generated images as Real (1).
    gan_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(tf.ones_like(fake_output), fake_output)

    # 2. PCC Loss (Structural Similarity)
    pcc = pcc_loss(real_img, fake_img)

    # Weighted Sum
    total_loss = (LAMBDA_ADV * gan_loss) + (LAMBDA_PCC * pcc)
    return total_loss, gan_loss, pcc

def discriminator_loss(real_output, fake_output):
    """
    Discriminator Loss.
    Real images should be classified as 1; Fake images as 0.
    """
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

# ==============================================================================
# DATA LOADING PIPELINE
# ==============================================================================

def load_image_pair(input_file, label_file):
    """
    Reads and preprocesses image pairs.
    Normalization: [0, 255] -> [0.0, 1.0]
    """
    # Load Input (Speckle)
    input_img = tf.io.read_file(input_file)
    input_img = tf.image.decode_bmp(input_img, channels=1)
    input_img = tf.image.resize(input_img, [IMG_HEIGHT, IMG_WIDTH])
    input_img = tf.cast(input_img, tf.float32) / 255.0

    # Load Label (Ground Truth)
    label_img = tf.io.read_file(label_file)
    label_img = tf.image.decode_bmp(label_img, channels=1)
    label_img = tf.image.resize(label_img, [IMG_HEIGHT, IMG_WIDTH])
    label_img = tf.cast(label_img, tf.float32) / 255.0

    return input_img, label_img

def get_dataset(input_dir, label_dir, batch_size):
    """
    Creates a highly optimized tf.data.Dataset pipeline.
    """
    # Verify file existence
    input_files = sorted(glob.glob(os.path.join(input_dir, '*.bmp')))
    label_files = sorted(glob.glob(os.path.join(label_dir, '*.bmp')))

    if len(input_files) == 0:
        print(f"Error: No images found in {input_dir}. Please check the path.")

    dataset = tf.data.Dataset.from_tensor_slices((input_files, label_files))
    dataset = dataset.map(load_image_pair, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=500)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# ==============================================================================
# MODEL ARCHITECTURES
# ==============================================================================

def build_unet_generator(input_shape=(256, 256, 1)):
    """
    Defines a 5-level U-Net Generator.
    """
    inputs = Input(shape=input_shape)

    # Convolutional Block Helper
    def conv_block(x, filters, apply_bn=True):
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        if apply_bn: x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        if apply_bn: x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    # --- Encoder (Downsampling) ---
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

    # --- Decoder (Upsampling) ---
    u7 = layers.Conv2DTranspose(256, 3, strides=2, padding='same')(c6)
    u7 = layers.Concatenate()([u7, c5])
    c7 = conv_block(u7, 256)

    u8 = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(c7)
    u8 = layers.Concatenate()([u8, c4])
    c8 = conv_block(u8, 128)

    u9 = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(c9) # Fixed index continuity
    u9 = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(c8)
    u9 = layers.Concatenate()([u9, c3])
    c9 = conv_block(u9, 64)

    u10 = layers.Conv2DTranspose(32, 3, strides=2, padding='same')(c9)
    u10 = layers.Concatenate()([u10, c2])
    c10 = conv_block(u10, 32)

    u11 = layers.Conv2DTranspose(16, 3, strides=2, padding='same')(c10)
    u11 = layers.Concatenate()([u11, c1])
    c11 = conv_block(u11, 16)

    # Output Layer
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c11)

    return Model(inputs=inputs, outputs=outputs, name="Generator")

def build_discriminator(input_shape=(256, 256, 1)):
    """
    Defines the Discriminator.
    Input: Concatenation of [Target Image, Condition Image].
    """
    img_input = Input(shape=input_shape, name='target_img')     # Ground Truth or Generated
    cond_input = Input(shape=input_shape, name='condition_img') # Speckle Input

    # Concatenate along channel axis
    combined = layers.Concatenate()([img_input, cond_input])

    # Downsampling blocks with LeakyReLU
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

    # Global Average Pooling for scalar output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    return Model(inputs=[img_input, cond_input], outputs=x, name="Discriminator")

# ==============================================================================
# GAN TRAINING CLASS
# ==============================================================================

class SpeckleGAN(Model):
    def __init__(self, generator, discriminator):
        super(SpeckleGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, g_optimizer, d_optimizer):
        super(SpeckleGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    @tf.function
    def train_step(self, data):
        speckle, real_gt = data

        # --- Train Discriminator ---
        with tf.GradientTape() as tape:
            # Generate fake images
            fake_gt = self.generator(speckle, training=True)

            # Get Discriminator predictions
            real_pred = self.discriminator([real_gt, speckle], training=True)
            fake_pred = self.discriminator([fake_gt, speckle], training=True)

            # Calculate D loss
            d_loss = discriminator_loss(real_pred, fake_pred)

        # Apply gradients to D
        d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        # --- Train Generator ---
        with tf.GradientTape() as tape:
            # Generate fake images
            fake_gt = self.generator(speckle, training=True)

            # Discriminator predictions for G update
            fake_pred = self.discriminator([fake_gt, speckle], training=True)

            # Calculate G loss (Adversarial + PCC)
            total_g_loss, adv_loss, pcc = generator_loss(fake_pred, real_gt, fake_gt)

        # Apply gradients to G
        g_grads = tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": total_g_loss, "pcc": pcc}

# ==============================================================================
# MAIN EXECUTION FLOW
# ==============================================================================

if __name__ == '__main__':
    # 1. Directory Setup
    for folder in [CHECKPOINT_DIR, LOG_DIR, EVAL_DIR]:
        os.makedirs(folder, exist_ok=True)

    # 2. Build Models
    print("Building models...")
    generator = build_unet_generator()
    discriminator = build_discriminator()

    # 3. Transfer Learning (Freezing Layers)
    if PRETRAINED_GENERATOR_PATH is not None and os.path.exists(PRETRAINED_GENERATOR_PATH):
        print(f"Loading pretrained weights from: {PRETRAINED_GENERATOR_PATH}")
        # Load weights partially if possible
        generator.load_weights(PRETRAINED_GENERATOR_PATH, by_name=True)

        # Freezing logic: Freeze the Encoder part (first 80% of layers)
        freeze_ratio = 0.8
        num_layers = len(generator.layers)
        num_freeze = int(num_layers * freeze_ratio)

        print(f"Freezing the first {num_freeze} layers (Encoder)...")
        for i, layer in enumerate(generator.layers):
            if i < num_freeze:
                layer.trainable = False

        print("Pretrained layers frozen.")

    # 4. Compile GAN
    print("Compiling GAN...")
    gan = SpeckleGAN(generator, discriminator)
    gan.compile(
        g_optimizer=Adam(LR_GENERATOR, beta_1=0.5),
        d_optimizer=Adam(LR_DISCRIMINATOR, beta_1=0.5)
    )

    # 5. Prepare Dataset
    print("Initializing data pipeline...")
    train_ds = get_dataset(DATASET_INPUT_DIR, DATASET_LABEL_DIR, BATCH_SIZE)

    # 6. Define Callbacks
    # Checkpoint: Save model weights every epoch
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(CHECKPOINT_DIR, "generator_epoch_{epoch:03d}.h5"),
        save_weights_only=True,
        save_freq='epoch'
    )

    # Visualization: Save comparison images periodically
    class VisualizationCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 5 == 0: # Visualize every 5 epochs
                # Take one batch for visualization
                for speckle, real in train_ds.take(1):
                    fake = self.model.generator(speckle, training=False)

                    # Process the first image in the batch
                    s = speckle[0].numpy().squeeze()
                    r = real[0].numpy().squeeze()
                    f = fake[0].numpy().squeeze()

                    # Plot comparison
                    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                    ax[0].imshow(s, cmap='gray'); ax[0].set_title('Speckle Input')
                    ax[0].axis('off')
                    ax[1].imshow(f, cmap='gray'); ax[1].set_title(f'Reconstruction (Ep {epoch+1})')
                    ax[1].axis('off')
                    ax[2].imshow(r, cmap='gray'); ax[2].set_title('Ground Truth')
                    ax[2].axis('off')

                    save_path = os.path.join(EVAL_DIR, f"epoch_{epoch+1}.png")
                    plt.savefig(save_path, bbox_inches='tight')
                    plt.close()
                    break

    # 7. Start Training
    print(f"Starting training for {EPOCHS} epochs...")
    gan.fit(
        train_ds,
        epochs=EPOCHS,
        callbacks=[ckpt_cb, VisualizationCallback()]
    )
    print("Training process completed successfully.")