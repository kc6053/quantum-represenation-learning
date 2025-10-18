
import torch
import os
from torchvision.utils import save_image

from .config import DEVICE, IMG_SIZE, IMG_CHANNELS, OUTPUT_DIR, CHECKPOINT_DIR, EPOCHS
from .model import UNet
from .diffusion_utils import sample

def sample_and_save(num_images=16):
    """
    Loads a trained model checkpoint, generates new images, and saves them.
    """
    # --- 1. Setup ---
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- 2. Load Model ---
    # Instantiate the model
    model = UNet()
    model.to(DEVICE)
    
    # Construct the path to the final checkpoint
    # NOTE: Adjust the epoch number if you want to load a different checkpoint.
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"ddpm_mnist_epoch_{EPOCHS}.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please run the training script first using 'python -m diffusion.main'")
        return

    print(f"Loading checkpoint from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval() # Set the model to evaluation mode
    print("Model loaded.")

    # --- 3. Generate Images ---
    print(f"Generating {num_images} new images...")
    # The sample function returns a list of images from each step of the reverse process.
    # We are interested in the final denoised images, so we take the last item.
    generated_images_steps = sample(model, image_size=IMG_SIZE, batch_size=num_images, channels=IMG_CHANNELS, device=DEVICE)
    final_images = generated_images_steps[-1]
    
    # --- 4. Save Images ---
    # Denormalize the images from [-1, 1] to [0, 1] before saving
    final_images = (final_images + 1) * 0.5
    
    # Save the images as a grid
    save_path = os.path.join(OUTPUT_DIR, "generated_images.png")
    save_image(final_images, save_path, nrow=4) # Arrange images in a 4-column grid
    
    print(f"Successfully saved {num_images} generated images to {save_path}")

if __name__ == '__main__':
    sample_and_save()
