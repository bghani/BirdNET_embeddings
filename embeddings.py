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

# Define the directory where the BirdNET-Analyzer module is located
MODULE_DIR = "/home/ubuntu/burooj/BirdNET-Analyzer"
sys.path.append(MODULE_DIR)

# Import necessary functions and classes from the BirdNET-Analyzer
from util import embed_sample, BirdNet

# Define the paths and parameters
EMBEDDING_MODEL_PATH = '/home/ubuntu/burooj/BirdNET-Analyzer/checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite'
SOUND_FILE = '/home/ubuntu/burooj/embeddings-baseline/combined_normalized.wav'
SOURCE_DIR = '/data/burooj/data_Burooj'
TARGET_DIR = '/data/burooj/ines-data/data_Burooj_BirdNETembeddings_10'
SAMPLE_RATE = 48000

# Load the BirdNET model
embedding_model = BirdNet(48000, EMBEDDING_MODEL_PATH)

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
