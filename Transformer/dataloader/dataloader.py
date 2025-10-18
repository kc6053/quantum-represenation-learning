
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from Transformer.config import *

# --- Special Tokens ---
# These tokens are used to signify padding, start of sequence, and end of sequence.
PAD_TOKEN = 0  # Used to pad sequences to the same length.
SOS_TOKEN = 1  # "Start of Sequence" token.
EOS_TOKEN = 2  # "End of Sequence" token.

class Seq2SeqDataset(Dataset):
    """
    A toy dataset that generates random sequences of integers and their reversed counterparts.
    This is a simple sequence-to-sequence task used to demonstrate the Transformer model.
    """
    def __init__(self, num_samples, max_seq_length, vocab_size):
        self.num_samples = num_samples
        self.max_seq_length = max_seq_length
        # Vocab size should be greater than 3 to account for special tokens
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Generate a random sequence of integers
        seq_length = np.random.randint(3, self.max_seq_length)
        # Integers are from 3 up to vocab_size to avoid collision with special tokens
        seq = np.random.randint(3, self.vocab_size, seq_length)
        
        # The source is the original sequence
        src = torch.LongTensor(seq)
        # The target is the reversed sequence. .copy() is important to avoid a negative stride issue.
        tgt = torch.LongTensor(np.flip(seq).copy())

        # 2. Add Start of Sequence (SOS) and End of Sequence (EOS) tokens
        src = torch.cat([torch.LongTensor([SOS_TOKEN]), src, torch.LongTensor([EOS_TOKEN])])
        tgt = torch.cat([torch.LongTensor([SOS_TOKEN]), tgt, torch.LongTensor([EOS_TOKEN])])

        # 3. Pad the sequences to the maximum length
        # We add 2 to max_seq_length to account for the SOS and EOS tokens.
        src_padded = torch.full((self.max_seq_length + 2,), PAD_TOKEN, dtype=torch.long)
        src_padded[:len(src)] = src

        tgt_padded = torch.full((self.max_seq_length + 2,), PAD_TOKEN, dtype=torch.long)
        tgt_padded[:len(tgt)] = tgt

        return src_padded, tgt_padded

def create_dataloaders(batch_size, num_samples, max_seq_length, vocab_size):
    """
    Factory function to create a DataLoader for our Seq2SeqDataset.
    """
    dataset = Seq2SeqDataset(num_samples, max_seq_length, vocab_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

if __name__ == '__main__':
    # This block demonstrates how to use the dataloader and what the data looks like.
    print("--- Dataloader Example ---")
    dataloader = create_dataloaders(BATCH_SIZE, 1000, MAX_SEQ_LENGTH, SRC_VOCAB_SIZE)
    # Get one batch of data
    src, tgt = next(iter(dataloader))
    
    print(f"Batch size: {src.shape[0]}")
    print(f"Sequence length: {src.shape[1]}")
    print("\nExample source sequence (padded):")
    print(src[0])
    print("\nExample target sequence (padded, reversed):")
    print(tgt[0])
    print("\nNote: 0=PAD, 1=SOS, 2=EOS")
