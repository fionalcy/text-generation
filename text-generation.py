import tensorflow as tf
import config

# Load the pre-trained model for retraining
one_step_reloaded = tf.saved_model.load(f'{config.folder}-ready-{config.save_model}')
print(type(one_step_reloaded))

states = None

if len(config.seed_text) < 1:
  seed_text = " "
else:
  seed_text = config.seed_text

next_char = tf.constant([seed_text])
result = [next_char]

# Generate a sequence of characters based on the loaded model
for n in range(config.num_characters_generate):
  next_char, states = one_step_reloaded.generate_one_step(next_char, states=states)
  result.append(next_char)

# Convert the generated character sequence to a string
generated_text = tf.strings.join(result)[0].numpy().decode("utf-8")

# Print the generated text to the console
print(generated_text)

# Save the generated text to an output file
with open(f"{config.folder}/{config.text_generate_output}", "w") as f:
  f.write(generated_text)

# Provide a message indicating where the output is saved
print(f"Output is saved at : {config.folder}/{config.text_generate_output} ")
