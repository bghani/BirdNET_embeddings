# -*- coding: utf-8 -*-
# @Time    : 23/05/23 17:00 
# @Author  : Burooj Ghani
# @Affiliation  : Naturalis Biodiversity Center
# @Email   : burooj.ghani at naturalis.nl

# Parts of this code are adapted from: https://github.com/google-research/perch

import numpy as np
import librosa
import os
import tensorflow as tf
import tempfile
from etils import epath
import logging
from typing import Any, List
import dataclasses
import glob
import json


@dataclasses.dataclass    
class EmbeddingModel:
  """Wrapper for a model which produces audio embeddings.

  Attributes:
    sample_rate: Sample rate in hz.
  """

  sample_rate: int

  def embed(self, audio_array: np.ndarray) -> np.ndarray:
    """Create evenly-spaced embeddings for an audio array.

    Args:
      audio_array: An array with shape [Time] containing unit-scaled audio.

    Returns:
      An InferenceOutputs object.
    """
    raise NotImplementedError

  def batch_embed(self, audio_batch: np.ndarray) -> np.ndarray:
    """Embed a batch of audio."""
    outputs = []
    for audio in audio_batch:
      outputs.append(self.embed(audio))
    if outputs[0].embeddings is not None:
      embeddings = np.stack([x.embeddings for x in outputs], axis=0)
    else:
      embeddings = None

    return embeddings
    
  def frame_audio(
      self,
      audio_array: np.ndarray,
      window_size_s: "float | None",
      hop_size_s: float,
  ) -> np.ndarray:
    """Helper function for framing audio for inference."""
    if window_size_s is None or window_size_s < 0:
      return audio_array[np.newaxis, :]
    frame_length = int(window_size_s * self.sample_rate)
    hop_length = int(hop_size_s * self.sample_rate)
    # Librosa frames as [frame_length, batch], so need a transpose.
    framed_audio = librosa.util.frame(audio_array, frame_length, hop_length).T
    return framed_audio

@dataclasses.dataclass
class BirdNET(EmbeddingModel):
  """Wrapper for BirdNET models.

  Attributes:
    model_path: Path to the saved model checkpoint or TFLite file.
    class_list_name: Name of the BirdNET class list.
    window_size_s: Window size for framing audio in samples.
    hop_size_s: Hop size for inference.
    num_tflite_threads: Number of threads to use with TFLite model.
    target_class_list: If provided, restricts logits to this ClassList.
    model: The TF SavedModel or TFLite interpreter.
    tflite: Whether the model is a TFLite model.
    class_list: The loaded class list.
  """

  model_path: str
  class_list_name: str = 'birdnet_v2_1'
  window_size_s: float = 3.0
  hop_size_s: float = 3.0
  num_tflite_threads: int = 16
  target_class_list: "namespace.ClassList | None" = None
  # The following are populated during init.
  model: "Any | None" = None
  tflite: bool = False
  class_list: "namespace.ClassList | None" = None

  def __post_init__(self):
    logging.info('Loading BirdNET model...')
    if self.model_path.endswith('.tflite'):
      self.tflite = True
      with tempfile.NamedTemporaryFile() as tmpf:
        model_file = epath.Path(self.model_path)
        model_file.copy(tmpf.name, overwrite=True)
        self.model = tf.lite.Interpreter(
            tmpf.name, num_threads=self.num_tflite_threads
        )
      self.model.allocate_tensors()
    else:
      self.tflite = False
      

  def embed_tflite(self, audio_array: np.ndarray) -> np.ndarray:
    """Create an embedding and logits using the BirdNET TFLite model."""
    input_details = self.model.get_input_details()[0]
    output_details = self.model.get_output_details()[0]
    embedding_idx = output_details['index'] - 1
    embeddings = []
    logits = []
    for audio in audio_array:
      self.model.set_tensor(
          input_details['index'], np.float32(audio)[np.newaxis, :]
      )
      self.model.invoke()
    
      embeddings.append(self.model.get_tensor(embedding_idx))
      logits.append(self.model.get_tensor(output_details['index']))
    # Create [Batch, 1, Features]
    embeddings = np.array(embeddings)
    logits = np.array(logits)
    
    return embeddings, logits
    

  def embed(self, audio_array: np.ndarray) -> np.ndarray:
    framed_audio = self.frame_audio(
        audio_array, self.window_size_s, self.hop_size_s
    )
    
    return self.embed_tflite(framed_audio)
    


def embed_sample(
    embedding_model: EmbeddingModel,
    sample: np.ndarray,
    data_sample_rate: int,
) -> np.ndarray:
  
  """Compute embeddings for an audio sample.

  Args:
    embedding_model: Inference model.
    sample: audio example.
    data_sample_rate: Sample rate of dataset audio.

  Returns:
    Numpy array containing the embeddeding.
  """
  try:
        if data_sample_rate > 0 and data_sample_rate != embedding_model.sample_rate:
            sample = librosa.resample(
                sample,
                data_sample_rate,
                embedding_model.sample_rate,
                res_type='polyphase',
            )

        audio_size = sample.shape[0]
        if hasattr(embedding_model, 'window_size_s'):
            window_size = int(
                embedding_model.window_size_s * embedding_model.sample_rate
            )
        if window_size > audio_size:
            pad_amount = window_size - audio_size
            front = pad_amount // 2
            back = pad_amount - front + pad_amount % 2
            sample = np.pad(sample, [(front, back)], 'constant')

        outputs = embedding_model.embed(sample)
        
        if outputs is not None:
        #embeds = outputs.embeddings.mean(axis=1).squeeze()
            embed = outputs[0].mean(axis=0).squeeze()
            logits = outputs[1].squeeze().squeeze()

        return embed, logits
        
  except:
        return None
  

def embed_files(sr, source_directory, target_directory, model, output_format):
    """
    Process sound files to compute and save their BirdNET embeddings.
    
    :param source_directory: Directory containing the .wav files
    :param target_directory: Directory where the embeddings will be saved
    :param model: The BirdNET model used for generating embeddings
    """
    # Get list of all sound files in source directory
    sound_files = glob.glob(os.path.join(source_directory, '*.wav')) + glob.glob(os.path.join(source_directory, '*.mp3')) + glob.glob(os.path.join(source_directory, '*.WAV'))

    # Check if the target directory exists, and create it if it doesn't
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)


    for sound_file in sound_files:
        # Load the sound file
        try:
          y, fs = librosa.load(sound_file, sr=sr, offset=0.0, res_type='kaiser_fast')
        except:
          continue

        # Compute the embedding
        embedding, _ = embed_sample(model, y, sr)

        # Determine the output .npy filename
        file_extension = os.path.splitext(sound_file)[1].lower()
        base_filename = os.path.basename(sound_file).replace(file_extension, '')
        
        if file_extension in ['.wav', '.WAV', '.mp3']:
          # Define the output file path
          if output_format == 'npy':
              output_file = os.path.join(target_directory, base_filename + '.npy')
              # Save the embedding as a .npy file
              np.save(output_file, embedding)
          elif output_format == 'json':
              output_file = os.path.join(target_directory, base_filename + '.json')
              # Convert the embedding to a list (if it's an array) and save as JSON
              with open(output_file, 'w') as f:
                  json.dump(embedding.tolist(), f)
          else:
              raise ValueError("Unsupported output format: " + output_format)