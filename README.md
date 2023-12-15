# Text Generation with TensorFlow

Text Generation with TensorFlow is a project that uses deep learning techniques to generate text sequences. This project leverages TensorFlow, a popular deep learning framework, and provides the ability to train models on input text data and generate creative text based on the learned patterns.

This script is heavily influenced by: 
- https://github.com/gsurma/text_predictor/ 
- https://www.tensorflow.org/text/tutorials/text_generation


## Table of Contents
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)

## Getting Started

To get started with text generation, follow these steps:

1. Clone this repository to your local machine: git clone https://git.arts.ac.uk/22045067/text-generation.git
2. Install the required dependencies. You can use `pip` to install them:
`pip install -r requirements.txt`
   (This project is based on Python3 Tensorflow2. If you experience trouble of installing Tensorflow==2.15 as specific in the requirements.txt. 
   Feel free to download any Tensorflow2)

3. Prepare your input text data (.txt). Create a text file and specify the folder and file path in the configuration.
4. Customize the model and training settings in the `config.py` file according to your preferences.

5. Run the text predict script:
`python3 text-predict.py` or `python text-predict.py` depends on your python setting
    This script encompasses functionalities such as word counting, dataset preparation, training text generation models, and generating text based on trained models.
    After running the script, the trained model would be saved as - [config.model_name].keras in the [config.folder]
6. To continue training the model - run `python3 retrain-model.py` , and have the appropriate configuration in `config.py`.
7. To use the model - run `python3 text-generation.py` , and have the appropriate configuration in `config.py`

## Configuration

In the `config.py` file, you can configure various settings for your text generation project:

### Project Settings
- `folder`: Directory where input files, output files, models, and TensorFlow board logs will be saved.

### Script Settings
- `input_file`: Path to the input text file.
- `preview`: Number of characters to preview from the input text (set to 0 for no preview).

### Wordcount Settings
- `output_text`: Set to `True` to output the word count to a text file.
- `output_terminal`: Set to `True` to display the word count in the terminal.
- `chinese`: Specify counting mode: `True` for Chinese-like languages, `False` for English-like languages.
- `output_file_name`: Name of the output text file for word count results.

### Dataset Settings
- `batch_size`: Number of examples per batch.
- `sequence_size`: Length of each example sequence.
- `buffer_size`: Buffer size for shuffling the dataset (should be greater than the number of elements).

### Model Settings
- `embedding_dim`: Number of embedding dimensions.
- `epochs`: Number of training epochs.
- `checkpoint_folder`: Directory for saving model checkpoints.

### Model Selection
- `rnn_units`: Number of units in the RNN layer.
- `model`: Choose between GRU and LSTM for the RNN model (Enter 'GRU' or 'LSTM').

### Logging Settings
- `logging_frequency`: Frequency of model logging (how often to print model output).
- `characters`: Number of characters to output in each logging step.
- `logging_file`: File for logging the generated text output.

### Saved Model Name
- `save_model`: Name of the saved model (will be saved in the data folder).

### Retrain Setting
- `retrain_epoch`: Number of epochs for retraining the model.

### Text Generation Setting
- `num_characters_generate`: Number of characters to generate.
- `seed_text`: " " Seed text for text generation (can be any text, at least a space).
- `text_generate_output`: Name of the output file for generated text.


## Data Preparation

Text data should be prepared in a plain text file. The project uses TensorFlow to preprocess the data, including converting characters to numerical IDs and creating datasets for training.

## Model Training

The project trains a custom-defined text generation model using TensorFlow. The model is compiled with loss and optimizer functions. Checkpoints are saved during training for later use.

- Model Select - GRU: https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU
- Model Select - LSTM:https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
