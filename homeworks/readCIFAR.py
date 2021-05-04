import numpy as np
import pickle

def readCIFAR(filename):
    with open(filename, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')

    # Get the raw images.
    raw_images = data[b'data']

    # Get the class-numbers for each image. Convert to numpy-array.
    y = np.array(data[b'labels'])

    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw_images, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, 3, 32, 32])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    X = images.reshape((images.shape[0], 3*32*32))
    return X, y

# Example usage: read batch 1 for training
X, y = readCIFAR("data_batch_1")
