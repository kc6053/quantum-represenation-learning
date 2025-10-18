
import torch
from .config import DEVICE, LEARNING_RATE, EPOCHS
from .model import UNet
from .dataloader import get_dataloader
from .train import train

def main():
    """
    Main function to run the DDPM training pipeline.
    """
    print(f"--- Starting DDPM Training on {DEVICE} ---")

    # 1. Get the dataloader
    print("Loading MNIST dataset...")
    dataloader = get_dataloader()
    print("Dataset loaded.")

    # 2. Initialize the U-Net model
    print("Initializing U-Net model...")
    model = UNet()
    print("Model initialized.")

    # 3. Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Start the training process
    train(model, dataloader, optimizer, EPOCHS)

if __name__ == '__main__':
    main()
