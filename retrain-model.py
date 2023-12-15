import tensorflow as tf
import config
import sys
import functions
import models
import datetime
import time

folder = config.folder
model_name = config.save_model # Define the model name to be used

# Decode text
try:
    text = open(f"{folder}/{config.input_file}", 'rb').read().decode(encoding='utf-8')
except FileNotFoundError:
    print(f"The folder you have entered cannot be found, please make sure it is in the correct folder with the correct name")
    sys.exit(0)

vocab = sorted(set(text))
ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
chars_from_ids = tf.keras.layers.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

# Training + Predicting
all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
sequences = ids_dataset.batch(config.batch_size, drop_remainder=True)
dataset = sequences.map(functions.split_input_target)

# Prepare the dataset
dataset = (
    dataset
    .shuffle(config.buffer_size)
    .batch(config.batch_size, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

# Define a callback for saving model checkpoints during training
checkpoint = tf.keras.callbacks.ModelCheckpoint(f"{config.folder}/{config.checkpoint_folder}/ckpt_{config.epochs}",
                                                save_weights_only=True)

# Load the pre-trained model for retraining
model_reloaded = tf.keras.models.load_model(f"{folder}/{model_name}.keras")

# Define a callback for TensorBoard logging
log_dir = f"{folder}/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Retrain the model for a specified number of epochs with logging frequency
for i in range(0, config.retrain_epoch, config.logging_frequency):
    # Fit the model to the dataset for a certain number of epochs
    print(f"This is Epoch {i}")
    history = model_reloaded.fit(dataset, epochs=config.logging_frequency, callbacks=[checkpoint, tensorboard_callback])

    # Create a one-step model for generating text
    one_step_model = models.OneStep(model_reloaded, chars_from_ids, ids_from_chars)

    # Text generation
    start = time.time()
    states = None
    next_char = tf.constant([' '])
    result = [next_char]

    for n in range(config.characters):  # number of characters to generate
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    end = time.time()

    # Print generated text and save it to the output file
    print(result[0].numpy().decode('utf-8'), '\n\n' + '_' * 80)
    with open(f"{folder}/{config.logging_file}", 'a') as f:
        f.write(f"Epoch: {i}\n")
        f.write(result[0].numpy().decode('utf-8'))
        f.write('\n\n')

    print('\nRun time:', end - start)

# Path to retrained model
model_reloaded.save(f"{folder}/{model_name}.keras")
print(model_reloaded.summary())
print(f"Finished Retraining - Epoch {config.retrain_epoch}")

# Save the one-step model for text generation
tf.saved_model.save(one_step_model, f'{folder}-ready-{model_name}')
print(f"Please do not delete the file /ready-{model_name}")
print(f"For text generation using the trained model, please run the script text-generate.py")
