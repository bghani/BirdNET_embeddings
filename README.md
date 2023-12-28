# Generate BirdNET Embeddings

This script generates BirdNET embeddings for all sound files located in a specified input directory. The embeddings are saved in a specified output directory, making it easy to process large batches of audio data for further analysis or machine learning applications.

## Prerequisites

Before you run this script, make sure you have the following installed:
- Python 3.6 or higher
- [librosa](https://librosa.org/doc/latest/index.html) for audio processing
- [numpy](https://numpy.org/) for numerical operations
- [BirdNET-Analyzer](https://github.com/kahst/BirdNET-Analyzer) module and its dependencies

## Installation

1. Clone this repository to your local machine.
2. Ensure that all required libraries are installed by running.


3. Set up the BirdNET-Analyzer according to its [documentation](https://github.com/kahst/BirdNET-Analyzer).

## Configuration

Before running the script, configure the following settings:

1. **BirdNET-Analyzer Directory**: Set the `BIRDNET_ANALYZER_DIR` environment variable to the path where the BirdNET-Analyzer module is located.
2. **Source and Target Directories**: Set the `SOURCE_DIR` and `TARGET_DIR` environment variables to your desired input and output directories, respectively.

Alternatively, you can modify these settings directly in the script or provide a `config.json` file with the necessary paths.

## Usage

To run the script, navigate to the directory containing the script and execute:

```bash
python embeddings.py
 ```

## Output 

The embeddings are saved as .npy files (numpy array format) in the specified output directory. Each file corresponds to an audio file from the input directory, containing the generated embedding.

## Contributing

Feel free to fork this repository and submit pull requests with improvements or report any issues you encounter.