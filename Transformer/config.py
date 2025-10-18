

import torch

# --- Model Hyperparameters ---
# These parameters define the architecture of the Transformer model.
# They are smaller than usual for efficient training on a CPU.
D_MODEL = 128           # The dimension of the embeddings and the model's hidden states.
N_HEAD = 4              # The number of heads in the multi-head attention mechanism.
NUM_ENCODER_LAYERS = 2  # The number of layers in the encoder.
NUM_DECODER_LAYERS = 2  # The number of layers in the decoder.
DIM_FEEDFORWARD = 512   # The dimension of the feed-forward network model.
DROPOUT = 0.1           # The dropout rate.
SRC_VOCAB_SIZE = 1000   # The size of the source vocabulary.
TGT_VOCAB_SIZE = 1000   # The size of the target vocabulary.

# --- Training Hyperparameters ---
# These parameters control the training process.
EPOCHS = 5              # The number of complete passes through the training dataset.
BATCH_SIZE = 16         # The number of samples processed before the model is updated.
LEARNING_RATE = 0.0001  # The step size at each iteration while moving toward a minimum of a loss function.
DEVICE = "cpu"          # The device to run the training on ('cuda' or 'cpu').

# --- Data Parameters ---
# These parameters are related to the data generation and processing.
MAX_SEQ_LENGTH = 50     # The maximum length of input and output sequences.
