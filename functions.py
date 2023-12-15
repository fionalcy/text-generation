import tensorflow as tf
from collections import defaultdict

# Define a function to convert IDs back to text
def text_from_ids(ids):
  return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

# Define a function to split a sequence into input and target text
def split_input_target(sequence):
    # The input text is the sequence up to the last character
    input_text = sequence[:-1]
    # The target text is the sequence from the second character onwards
    target_text = sequence[1:]
    return input_text, target_text

def word_count(text, chinese=True):
    test_dict = defaultdict(int)
    if chinese:
        for i in text:
            test_dict[i] += 1
    else:
        for i in text.split():
            test_dict[i] += 1
    return test_dict
