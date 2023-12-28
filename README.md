# Generate BirdNET Embeddings

This script generates BirdNET embeddings for all sound files located in a specified input directory (wav/mp3 files). The embeddings are saved in a specified output directory, making it easy to process large batches of audio data for further analysis or machine learning applications.


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

```
git clone https://github.com/kahst/BirdNET-Analyzer.git
```

This code has been tested with Python 3.9.12.

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

The embeddings are saved as .npy files by default (numpy array format) in the specified output directory, but can also be saved as json files by adding an argument to `embed_sample` function, in which case the embedding vectors will be saved as a list in respective json files. Each file corresponds to an audio file from the input directory, containing the generated embedding.

## Contributing

Feel free to fork this repository and submit pull requests with improvements or report any issues you encounter.