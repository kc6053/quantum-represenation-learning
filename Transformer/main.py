
import torch
import torch.nn as nn
from Transformer.config import *
from Transformer.model.model import TransformerModel
from Transformer.dataloader.dataloader import create_dataloaders
from Transformer.train.train import run_training

def main():
    """
    The main entry point for the training process.
    It orchestrates the creation of dataloaders, model, optimizer, and criterion,
    and then starts the training.
    """
    # --- 1. Create Dataloaders ---
    # Create a dataloader for the training set.
    train_dataloader = create_dataloaders(BATCH_SIZE, 1000, MAX_SEQ_LENGTH, SRC_VOCAB_SIZE)
    # Create a separate dataloader for the validation set.
    val_dataloader = create_dataloaders(BATCH_SIZE, 200, MAX_SEQ_LENGTH, SRC_VOCAB_SIZE) 

    # --- 2. Initialize Model ---
    # Instantiate the Transformer model with hyperparameters from the config file.
    model = TransformerModel(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(DEVICE)

    # --- 3. Setup Optimizer and Loss Function ---
    # Adam is a popular choice for an optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # CrossEntropyLoss is suitable for classification tasks.
    # `ignore_index=0` tells the loss function to ignore the padding token (PAD_TOKEN=0),
    # so that we don't penalize the model for its predictions on padding.
    criterion = nn.CrossEntropyLoss(ignore_index=0) 

    # --- 4. Start Training ---
    print("Starting training on device:", DEVICE)
    run_training(model, train_dataloader, val_dataloader, optimizer, criterion, EPOCHS, DEVICE)
    print("Training finished.")

if __name__ == '__main__':
    # This ensures the main function is called only when the script is executed directly.
    main()
