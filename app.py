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

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Define the directory where the BirdNET-Analyzer module is located
#MODULE_DIR = config.get('BIRDNET_ANALYZER_DIR', '/default/path/if/none/set')
#sys.path.append(MODULE_DIR)

# Import necessary functions and classes
from util import embed_sample, BirdNET, embed_files

# Set paths and other variables
EMBEDDING_MODEL_PATH = 'BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite') 
SAMPLE_RATE = config.get('SAMPLE_RATE', 48000)

# Load the BirdNET model
embedding_model = BirdNET(SAMPLE_RATE, EMBEDDING_MODEL_PATH)

# Streamlit app setup
st.title("Audio Embedding Generator and Visualization")

# Upload audio file
uploaded_files = st.file_uploader("Upload audio files", type=["wav", "mp3"], accept_multiple_files=True)

if uploaded_files:
    embeddings = []
    file_names = []

    # Process each uploaded audio file
    for uploaded_file in uploaded_files:
        # Save the uploaded file to a temporary path
        audio_path = os.path.join("temp", uploaded_file.name)
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Generate embeddings
        embedding = embed_sample(SAMPLE_RATE, audio_path, embedding_model)
        embeddings.append(embedding)
        file_names.append(uploaded_file.name)
        
        # Allow the user to download the embeddings
        npy_data = io.BytesIO()
        np.save(npy_data, embedding)
        npy_data.seek(0)
        st.download_button(
            label=f"Download Embeddings for {uploaded_file.name}",
            data=npy_data,
            file_name=f"{uploaded_file.name.split('.')[0]}_embedding.npy",
            mime="application/octet-stream"
        )
    
    # Convert embeddings to a numpy array for UMAP visualization
    embeddings_array = np.vstack(embeddings)

    # Visualize embeddings in UMAP space
    if len(embeddings) > 1:
        st.write("## UMAP Visualization")
        reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='euclidean')
        umap_embedding = reducer.fit_transform(embeddings_array)
        
        # Plot the UMAP visualization
        plt.figure(figsize=(10, 6))
        for i, name in enumerate(file_names):
            plt.scatter(umap_embedding[i, 0], umap_embedding[i, 1], label=name)
        
        plt.title("UMAP Projection of Audio Embeddings")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.legend(loc="best")
        st.pyplot(plt.gcf())
    else:
        st.write("Upload more than one audio file to visualize in UMAP space.")

else:
    st.write("Please upload audio files to generate embeddings.")
