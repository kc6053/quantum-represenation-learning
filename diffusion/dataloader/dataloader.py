
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .config import IMG_SIZE, BATCH_SIZE

def get_dataloader():
    """
    Prepares the MNIST dataset and returns a DataLoader.
    
    The function performs the following steps:
    1. Defines a series of transformations: 
       - Resize images to a consistent size (IMG_SIZE).
       - Convert images to PyTorch Tensors.
       - Normalize pixel values to the range [-1, 1], which is a common practice for training diffusion models.
    2. Downloads the MNIST training dataset (if not already present).
    3. Creates a DataLoader to serve the data in batches.
    """
    
    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), # Resize images
        transforms.ToTensor(),                  # Convert image to a tensor
        transforms.Normalize((0.5,), (0.5,))    # Normalize to [-1, 1] range
    ])
    
    # Download and load the training data
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    
    # Create the DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4, # Use multiple subprocesses to load the data
        pin_memory=True # For faster data transfer to the GPU (if available)
    )
    
    return dataloader

if __name__ == '__main__':
    # This block demonstrates how to use the dataloader and inspect the data.
    print("--- Dataloader Example ---")
    dataloader = get_dataloader()
    # Get one batch of images and labels
    images, labels = next(iter(dataloader))
    
    print(f"Batch of images shape: {images.shape}") # Should be [BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE]
    print(f"Batch of labels shape: {labels.shape}")
    print(f"Image pixel value range: min={images.min()}, max={images.max()}") # Should be close to [-1, 1]
