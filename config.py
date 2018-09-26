import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

corpus_loc = 'data/xiaohuangji50w_nofenci.conv'

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 4000
print_every = 1
save_every = 500
workers = 1
max_len = 10  # Maximum sentence length to consider
min_word_freq = 3    # Minimum word count threshold for trimming

# Configure models
model_name = 'cb_model'
attn_model = 'dot'
start_epoch = 0
epochs = 120
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
checkpoint_iter = 4000

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

