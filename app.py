# -*- coding: utf-8 -*-
# @Time    : 23/05/23 15:00 
# @Author  : Burooj Ghani
# @Affiliation  : Naturalis Biodiversity Center
# @Email   : burooj.ghani at naturalis.nl

import os
import sys
import json
import streamlit as st
import numpy as np
import umap
import matplotlib.pyplot as plt
import io

# Load configuration
#with open('config.json', 'r') as config_file:
#    config = json.load(config_file)

# Define the directory where the BirdNET-Analyzer module is located
#MODULE_DIR = config.get('BIRDNET_ANALYZER_DIR', '/default/path/if/none/set')
#sys.path.append(MODULE_DIR)

# Import necessary functions and classes
from util import embed_sample, BirdNET, embed_files

# Set paths and other variables
EMBEDDING_MODEL_PATH = 'BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite' 
SAMPLE_RATE = 48000

# Load the BirdNET model
embedding_model = BirdNET(SAMPLE_RATE, EMBEDDING_MODEL_PATH)

# Streamlit app setup
st.title("Audio Embedding Generator and Visualization")

# Upload audio file
uploaded_files = st.file_uploader("Upload audio files", type=["wav", "mp3"], accept_multiple_files=True)
if uploaded_files:
    embeddings_dict = {}
    # Add this before processing the uploaded files
    if not os.path.exists("temp"):
        os.makedirs("temp")
    # Process each uploaded audio file
    for uploaded_file in uploaded_files:
        # Save the uploaded file to a temporary path
        audio_path = os.path.join("temp", uploaded_file.name)
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Generate embeddings
        embedding = embed_sample(SAMPLE_RATE, audio_path, embedding_model)
        
        if embedding is not None:
            embeddings_dict[uploaded_file.name] = embedding.tolist()  # Convert numpy array to list for JSON serialization
        else:
            st.error(f"Failed to generate embedding for {uploaded_file.name}. Please check the file format and content.")

# Ensure the embeddings dictionary is not empty before creating the JSON
if embeddings_dict:
    # Create a JSON file with all embeddings
    json_data = json.dumps(embeddings_dict, indent=4)
    json_buffer = io.BytesIO()
    json_buffer.write(json_data.encode('utf-8'))
    json_buffer.seek(0)

    # Provide a single download button for the entire batch
    st.download_button(
        label="Download All Embeddings as JSON",
        data=json_buffer,
        file_name="embeddings_batch.json",
        mime="application/json"
    )
    
    st.write("Embeddings have been generated and stored in a single JSON file.")
else:
    st.warning("No valid embeddings were generated. Please check the uploaded files.")
