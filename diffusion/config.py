
import torch

# --- Training Hyperparameters ---
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Diffusion Hyperparameters ---
TIMESTEPS = 200  # Number of diffusion timesteps

# --- Model & Data Hyperparameters ---
IMG_SIZE = 32  # Input images will be resized to this size
IMG_CHANNELS = 1 # Number of channels in the input image (1 for MNIST)

# --- Paths ---
OUTPUT_DIR = "diffusion/outputs" # Directory to save generated images
CHECKPOINT_DIR = "diffusion/checkpoints" # Directory to save model checkpoints
