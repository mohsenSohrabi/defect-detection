# The number of classes in the classification task
NUM_CLASSES = 2

# The number of samples that will be propagated through the network simultaneously
BATCH_SIZE = 16

# The path to your dataset
DATASET_PATH = 'dataset/'

# The maximum length of the tokens. Any token length more than this will be truncated
TOKEN_MAX_LEN = 4096

# The checkpoint for the pretrained model
CHECKPOINT = "microsoft/codebert-base"

# LSTM parameters
# The number of expected features in the input
LSTM_INPUT_SIZE = TOKEN_MAX_LEN

# The number of features in the hidden state
LSTM_HIDDEN_SIZE = 256

# Number of recurrent layers
LSTM_NUM_LAYERS = 2

# Learning rate for the optimizer
LSTM_LEARNING_RATE = 0.01

# Number of epochs for training the model
LSTM_NUM_EPOCHS = 20

# Transformer parameters
# The number of expected features in the input
TRANSFORMER_D_MODEL = TOKEN_MAX_LEN

# The number of heads in the multiheadattention models
TRANSFORMER_NHEAD = 8

# The number of sub-encoder-layers in the transformer model
TRANSFORMER_NUM_LAYERS = 3

# Learning rate for the optimizer
TRANSFORMER_LEARNING_RATE = 0.005

# Number of epochs for training the model
TRANSFORMER_NUM_EPOCHS = 100

# CNN parameters
# The number of expected features in the input
CNN_D_MODEL = TOKEN_MAX_LEN

# Learning rate for the optimizer
CNN_LEARNING_RATE = 0.005

# Number of epochs for training the model
CNN_NUM_EPOCHS = 100
