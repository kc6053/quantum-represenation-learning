
import torch
import torch.nn as nn
from Transformer.config import *

def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Performs one full training epoch.
    """
    model.train()  # Set the model to training mode
    total_loss = 0

    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)

        # --- Teacher Forcing ---
        # For a seq2seq task, the decoder input is the target sequence shifted right.
        # The model is given the ground truth token as input for the next prediction.
        # `tgt_input` is `<SOS> token1 token2...`
        # `tgt_out` is `token1 token2... <EOS>`
        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        # --- Create Masks ---
        # src_padding_mask: to ignore padding tokens in the source sequence.
        # tgt_padding_mask: to ignore padding tokens in the target sequence.
        # tgt_mask: to prevent the decoder from looking at future tokens (causal masking).
        src_padding_mask = (src == 0).to(device) # Assuming PAD_TOKEN is 0
        tgt_padding_mask = (tgt_input == 0).to(device)
        tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)

        optimizer.zero_grad()

        # --- Forward Pass ---
        output = model(src, tgt_input, 
                       tgt_mask=tgt_mask, 
                       src_key_padding_mask=src_padding_mask, 
                       tgt_key_padding_mask=tgt_padding_mask)

        # --- Loss Calculation & Backpropagation ---
        # The output of the model is (batch_size, seq_len, vocab_size).
        # The criterion expects (N, C) where N is number of samples, C is number of classes.
        # So we reshape the output and target.
        loss = criterion(output.reshape(-1, output.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0

    # `torch.no_grad()` is used to disable gradient calculations, which saves memory and speeds up computation.
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)

            # Prepare decoder input and target output as in training
            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            # Create masks
            src_padding_mask = (src == 0).to(device)
            tgt_padding_mask = (tgt_input == 0).to(device)
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)

            # Forward pass
            output = model(src, tgt_input, 
                           tgt_mask=tgt_mask, 
                           src_key_padding_mask=src_padding_mask, 
                           tgt_key_padding_mask=tgt_padding_mask)

            # Calculate loss
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt_out.reshape(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)

def run_training(model, train_dataloader, val_dataloader, optimizer, criterion, epochs, device):
    """
    The main training loop that orchestrates the training over multiple epochs.
    """
    for epoch in range(epochs):
        # Run one epoch of training
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
        # Evaluate the model on the validation set
        val_loss = evaluate(model, val_dataloader, criterion, device)

        print(f"Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {val_loss:.3f}")
