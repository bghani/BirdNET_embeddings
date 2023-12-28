# -*- coding: utf-8 -*-
# @Time    : 23/05/23 15:00 
# @Author  : Burooj Ghani
# @Affiliation  : Naturalis Biodiversity Center
# @Email   : burooj.ghani at naturalis.nl

import os
import sys
import json

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Define the directory where the BirdNET-Analyzer module is located
MODULE_DIR = config.get('BIRDNET_ANALYZER_DIR', '/default/path/if/none/set')
sys.path.append(MODULE_DIR)

# Import necessary functions and classes from the BirdNET-Analyzer
from util import embed_sample, BirdNET, embed_files

# Set paths and other variables
EMBEDDING_MODEL_PATH = os.path.join(MODULE_DIR, 'checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite')
SOURCE_DIR = config.get('SOURCE_DIR', '/default/source/dir')
TARGET_DIR = config.get('TARGET_DIR', '/default/target/dir')
SAMPLE_RATE = config.get('SAMPLE_RATE', 48000)

# Load the BirdNET model
embedding_model = BirdNET(SAMPLE_RATE, EMBEDDING_MODEL_PATH)


if __name__ == "__main__":
    # Process the sound files and generate embeddings
    embed_files(SAMPLE_RATE, SOURCE_DIR, TARGET_DIR, embedding_model)
