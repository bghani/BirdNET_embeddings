import os
import numpy as np
import tensorflow as tf
import argparse
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2

def load_embeddings(directory):
    """
    Loads embeddings from specified directory and returns a tuple of data and labels.
    """
    data, labels = [], []
    for class_dir in os.listdir(directory):
        class_path = os.path.join(directory, class_dir)
        if os.path.isdir(class_path):
            for file in os.listdir(class_path):
                file_path = os.path.join(class_path, file)
                if file.endswith('.npy'):
                    embedding = np.load(file_path)
                elif file.endswith('.json'):
                    with open(file_path, 'r') as f:
                        embedding = np.array(json.load(f))
                else:
                    continue
                data.append(embedding)
                labels.append(class_dir)
    return np.array(data), np.array(labels)

def create_model(input_shape, num_classes, hidden_neurons, dropout):
    """
    Creates a neural network with an optional hidden layer.
    """
    model = tf.keras.models.Sequential()
    if hidden_neurons > 0:
        model.add(tf.keras.layers.Dense(hidden_neurons, input_shape=input_shape))
        model.add(tf.keras.layers.Dropout(0.5))  # Dropout layer
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train(directory, num_training_examples, hidden_neurons, dropout):
    # Load embeddings
    data, labels = load_embeddings(directory)

    # Convert labels to numerical format
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=num_training_examples, stratify=labels)

    # Create and compile the model
    num_classes = len(lb.classes_)
    model = create_model((data.shape[1],), num_classes, hidden_neurons, dropout)

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a NN on audio embeddings.')
    parser.add_argument('directory', type=str, help='Directory containing the embedding files')
    parser.add_argument('num_training_examples', type=int, help='Number of training examples per class')
    parser.add_argument('--hidden_neurons', type=int, default=0, help='Number of neurons in the hidden layer (0 for no hidden layer)')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout')
    args = parser.parse_args()

    # Call the train function with command-line arguments
    train(args.directory, args.num_training_examples, args.hidden_neurons, args.dropout)

