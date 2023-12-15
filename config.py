# Project Settings
folder = "text" # Directory where input files, output files, models, and TensorFlow board logs will be saved.

# Script Settings
input_file = 'text3.txt'  # Path to the input text file
preview = 250  # Number of characters to preview from the input text (set to 0 for no preview)

# Wordcount Settings
output_text = False # Set to True to output the word count to a text file.
output_terminal = True # Set to True to display the word count in the terminal.
chinese = True # Specify counting mode: True for Chinese-like languages, False for English-like languages.
output_file_name = "text-count-output.txt" # Name of the output text file for word count results.

# Dataset Settings
batch_size = 64  # Number of examples per batch
sequence_size = 100  # Length of each example sequence
buffer_size = 10000  # Buffer size for shuffling the dataset (should be greater than the number of elements)

# Initial Model Training Settings (for text-predict.py)
embedding_dim = 256  # Number of embedding dimensions
epochs = 10  # Number of training epochs
checkpoint_folder = f"gru-training_checkpoints_v1"  # Directory for saving model checkpoints

# Model Selection
# Choose between LSTM or GRU for the RNN model
# LSTM: 2, 4, 6.. units (original parameters, can be adjusted)
# GRU: 1024.. units (original parameters, can be adjusted)
rnn_units = 1024  # Number of units in the RNN layer
model = "GRU"  # Choose between GRU and LSTM (Enter 'GRU' or 'LSTM')

# Logging Settings
logging_frequency = 1  # Frequency of model logging (how often to print model output)
characters = 250  # Number of characters to output in each logging step
logging_file = 'gru-output1.txt'  # File for logging the generated text output

# Saved model name (this will be saved in the data folder)
save_model = "model-test-GRU"

# Retrain Setting (for retrain-model.py only)
retrain_epoch = 30 # Number of epochs for retraining the model.

# Text Generation Setting
num_characters_generate = 1000 # Number of characters to generate.
seed_text = " " # Seed text for text generation (can be any text, at least a space).
text_generate_output = "text-generate.txt" # Name of the output file for generated text.
