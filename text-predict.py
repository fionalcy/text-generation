import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import time
import datetime
import config  # Import custom configuration settings
import functions  # Import custom functions
import models  # Import custom model definitions

# Configuration and Hyperparameters

folder = config.folder
input_file = config.input_file
batch_size = config.batch_size
buffer_size = config.buffer_size
embedding_dim = config.embedding_dim
rnn_units = config.rnn_units
all_epochs = config.epochs
preview = config.preview
checkpoint_folder = config.checkpoint_folder
model_select = config.model
logging_frequency = config.logging_frequency
logging_characters = config.characters
output_file = config.logging_file
model_name = config.save_model

# Decode text
try:
    text = open(f"{folder}/{input_file}", 'rb').read().decode(encoding='utf-8')
except FileNotFoundError:
    print(f"The folder you have entered cannot be found, please make sure it is in the correct folder with the correct name")
    sys.exit(0)

print(f'Length of text: {len(text)} characters')

# Preview Text
if preview > 0:
    print("Text Preview")
    print(text[:preview])

# Vocabulary Creation
vocab = sorted(set(text))
print(f'{len(vocab)} unique characters')

# Printing text count
output_to_file = config.output_text
output_to_terminal = config.output_terminal
chinese = config.chinese
output_file_name = config.output_file_name

test_dict = functions.word_count(text, chinese=chinese)

if output_to_file:
    with open(f"{folder}/{output_file_name}", 'w') as f:
        f.write(str(sorted(test_dict.items(), key=lambda k_v: k_v[1], reverse=True)))

if output_to_terminal:
    print(sorted(test_dict.items(), key=lambda k_v: k_v[1], reverse=True))

# Processing the text
# Vectorize the text
# Translating text to id and id to text
ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
chars_from_ids = tf.keras.layers.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

# Training + Predicting
all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
sequences = ids_dataset.batch(batch_size, drop_remainder=True)
dataset = sequences.map(functions.split_input_target)

# Prepare the dataset
dataset = (
    dataset
    .shuffle(buffer_size)
    .batch(batch_size, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

# Model
vocab_size = len(ids_from_chars.get_vocabulary())

# Create the custom defined model
model = models.MyModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    model_sel=model_select)

# Training
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss)

# Directory and naming for model checkpoints
checkpoint_dir = f'./{folder}/{checkpoint_folder}'
checkpoint_prefix = os.path.join(checkpoint_dir, f"ckpt_{all_epochs}")
print(f"The check point is saved in {folder}/{checkpoint_folder}/ckpt_{all_epochs}")
print(f"Please use ckpt_{all_epochs} in the config file if you need to continue training")

# Callback to save model weights during training
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

# Callback to log model training loss during training
csv_file_path = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{config.logging_loss}"
print(f"Logger Path:{os.path.join(config.folder, csv_file_path)}")
logger_path = os.path.join(config.folder, csv_file_path)
csv_logger = tf.keras.callbacks.CSVLogger(logger_path, append=True)

# Callback for the Tensorboard
log_dir = f"{folder}/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Training Loop
for i in range(0, all_epochs, logging_frequency):
    # Fit the model to the dataset for a certain number of epochs
    print(f"This is Epoch {i}")
    history = model.fit(dataset, epochs=logging_frequency, callbacks=[checkpoint_callback, tensorboard_callback, csv_logger])
    # Create a one-step model for generating text
    one_step_model = models.OneStep(model, chars_from_ids, ids_from_chars)

    # Text generation
    start = time.time()
    states = None

    if len(config.seed_text) < 1:
        seed_text = " "
    else:
        seed_text = config.seed_text

    next_char = tf.constant([seed_text])
    result = [next_char]

    for n in range(logging_characters):  # number of characters to generate
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    end = time.time()

    # Print generated text and save it to the output file
    print(result[0].numpy().decode('utf-8'), '\n\n' + '_' * 80)
    with open(f"{folder}/{output_file}", 'a') as f:
        f.write(f"Epoch: {i}\n")
        f.write(result[0].numpy().decode('utf-8'))
        f.write('\n\n')

    print('\nRun time:', end - start)

# Print model summary
print(model.summary())

try:
    model.save(f"{folder}/{model_name}.keras")
    print(f"Please do not delete the model: {model_name}.keras")
except NotImplementedError:
    model.save(f"{folder}/{model_name}",save_format='tf')
    print(f"Please do not delete the model: {model_name}")

print(f"If you want to retrain this model: continue with the retrain-model.py")


# Save the one-step model for text generation
try:
    tf.saved_model.save(one_step_model, f'{folder}-ready-{model_name}')
    print(f"Please do not delete the file /ready-{model_name}")
    print(f"For text generation using the trained model, please run the script text-generate.py")
except Exception as e:
    print(e, "cannot save one-step model")

# Save training loss to image
df = pd.read_csv(os.path.join(config.folder, csv_file_path))
df[['loss']].plot()
plt.xlabel('epochs')
plt.title('Training loss')
image_path = os.path.join(config.folder, f'{csv_file_path.split(".")[-2]}.png')
plt.savefig(image_path)
print(f"Training loss is save at {image_path}")

