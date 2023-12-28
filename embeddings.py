# -*- coding: utf-8 -*-
# @Time    : 23/05/23 15:00 
# @Author  : Burooj Ghani
# @Affiliation  : Naturalis Biodiversity Center
# @Email   : burooj.ghani at naturalis.nl

import os
import sys
import numpy as np
import glob
import librosa
import json

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Define the directory where the BirdNET-Analyzer module is located
MODULE_DIR = config.get('BIRDNET_ANALYZER_DIR', '/default/path/if/none/set')
sys.path.append(MODULE_DIR)

# Import necessary functions and classes from the BirdNET-Analyzer
from util import embed_sample, BirdNET

EMBEDDING_MODEL_PATH = os.path.join(MODULE_DIR, 'checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite')
SOURCE_DIR = config.get('SOURCE_DIR', '/default/source/dir')
TARGET_DIR = config.get('TARGET_DIR', '/default/target/dir')
SAMPLE_RATE = config.get('SAMPLE_RATE', 48000)

# Load the BirdNET model
embedding_model = BirdNET(SAMPLE_RATE, EMBEDDING_MODEL_PATH)

def process_files(source_directory, target_directory, model):
    """
    Process sound files to compute and save their BirdNET embeddings.
    
    :param source_directory: Directory containing the .wav files
    :param target_directory: Directory where the embeddings will be saved
    :param model: The BirdNET model used for generating embeddings
    """
    # Get list of all sound files in source directory
    sound_files = glob.glob(os.path.join(source_directory, '*.wav')) + glob.glob(os.path.join(source_directory, '*.mp3')) + glob.glob(os.path.join(source_directory, '*.WAV'))


    for sound_file in sound_files:
        # Load the wav file
        y, fs = librosa.load(sound_file, sr=SAMPLE_RATE, offset=0.0, res_type='kaiser_fast')

        # Compute the embedding
        embedding, _ = embed_sample(model, y, SAMPLE_RATE)

        # Determine the output .npy filename
        file_extension = os.path.splitext(sound_file)[1].lower()
        if file_extension in ['.wav', '.WAV', '.mp3']:
            npy_file = os.path.join(target_directory, os.path.basename(sound_file).replace(file_extension, '.npy'))

            # Save the embedding
            np.save(npy_file, embedding)

if __name__ == "__main__":
    # Process the sound files and generate embeddings
    process_files(SOURCE_DIR, TARGET_DIR, embedding_model)
