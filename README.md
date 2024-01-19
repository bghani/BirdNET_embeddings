
## Overview
This repository features scripts for generating BirdNET embeddings from audio files and subsequently training a neural network on these embeddings. The workflow implemented in these scripts follows the pipeline outlined in the study "Global birdsong embeddings enable superior transfer learning for bioacoustic classification," as detailed in [Nature Scientific Reports](https://www.nature.com/articles/s41598-023-49989-z).

### Embedding Generation

## Installation

1. **System Requirements:**
   - For Unix/Linux systems, install FFmpeg by running:
     ```
     sudo apt-get install ffmpeg
     ```

2. **Clone the Repository:**
   - Clone this repository to your local machine using:
     ```
     git clone https://github.com/bghani/BirdNET_embeddings.git
     ```

3. **Create a Virtual Environment (Optional but recommended):**
   - Navigate to the cloned directory and create a virtual environment:
     ```bash
     python3 -m venv embeddings
     ```
   - Activate the virtual environment:
     ```bash
     source embeddings/bin/activate  # Unix/Linux/MacOS
     embeddings\Scripts\activate  # Windows
     ```

4. **Install Python Dependencies:**
   - Ensure that you have a `requirements.txt` file in your project directory that lists all the necessary Python packages.
   - Install all required libraries by running:
     ```
     pip install -r requirements.txt
     ```

5. **Clone the BirdNET-Analyzer repo:**
   - Clone this repository to your local machine using:
     ```
     git clone https://github.com/kahst/BirdNET-Analyzer.git
     ```

This code has been tested with Python 3.9.12.


## Usage for computing embeddings

The script for generating embeddings processes all sound files (wav/mp3) located in a specified source directory. It uses the BirdNET model to generate embeddings for each audio file, which are then saved in a specified target directory. This functionality facilitates the processing of large batches of audio data for further analysis or machine learning applications.

### Configuration

Before running the script, configure the following settings:

1. **BirdNET-Analyzer Directory**: Set the `BIRDNET_ANALYZER_DIR` environment variable to the path where the BirdNET-Analyzer module is located.
2. **Source and Target Directories**: Set the `SOURCE_DIR` and `TARGET_DIR` environment variables to your desired input and output directories, respectively.

Alternatively, you can modify these settings directly in the script or provide a `config.json` file with the necessary paths.

To run the script, navigate to the directory containing the script and execute:

```bash
python embeddings.py
 ```

The embeddings are saved as .npy files by default (numpy array format) in the specified target directory, but can also be saved as JSON files by adding an argument to `embed_files` function, in which case the embedding vectors will be saved as a lists in respective JSON files. Each file corresponds to an audio file from the source directory, containing the generated embedding.


## Usage for training a feed forward neural network using the computed embeddings

This script, `train.py`, is designed to train a neural network on embeddings. It supports creating a model with an optional hidden layer and includes dropout for regularization. The script can process embeddings in both `.npy` (NumPy array format) and `.json` format. 

### Arguments

- `directory`: (Required) The directory containing the subdirectories for embedding files for different classes. Each class's embeddings should be in a separate subdirectory.
- `num_training_examples`: (Required) The number of training examples to use per class for training; the rest of the examples in the subdirectories will be used for testing.
- `--hidden_neurons`: (Optional) The number of neurons in the hidden layer. If set to 0 (default), no hidden layer is used.
- `--dropout`: (Optional) The dropout rate for regularization. Default is 0.5. Dropout layer is only added if there is a hidden layer.

### Running the Script
To run the script, use the following command structure (without the optional arguments the model will train as a single-layer perceptron):

```bash
python train.py <directory> <num_training_examples> [--hidden_neurons <hidden_neurons>] [--dropout <dropout_rate>]
 ```

## Contributing

Feel free to fork this repository and submit pull requests with improvements or report any issues you encounter.