import random

import numpy as np
import matplotlib.pyplot as plt


# load test set data
def read_test_set():
    # Reading The Test Set
    test_images_file = open('t10k-images.idx3-ubyte', 'rb')
    test_images_file.seek(4)

    test_labels_file = open('t10k-labels.idx1-ubyte', 'rb')
    test_labels_file.seek(8)

    num_of_test_images = int.from_bytes(test_images_file.read(4), 'big')
    test_images_file.seek(16)

    test_set = []
    for n in range(num_of_test_images):
        image = np.zeros((784, 1))
        for i in range(784):
            image[i] = int.from_bytes(test_images_file.read(1), 'big') / 256

        label_value = int.from_bytes(test_labels_file.read(1), 'big')
        label = np.zeros((10, 1))
        label[label_value, 0] = 1

        test_set.append((image, label))
    return test_set


# load training set data
def read_training_set():
    # Reading The Train Set
    train_images_file = open('train-images.idx3-ubyte', 'rb')
    train_images_file.seek(4)
    num_of_train_images = int.from_bytes(train_images_file.read(4), 'big')
    train_images_file.seek(16)

    train_labels_file = open('train-labels.idx1-ubyte', 'rb')
    train_labels_file.seek(8)

    train_set = []
    for n in range(num_of_train_images):
        image = np.zeros((784, 1))
        for i in range(784):
            image[i, 0] = int.from_bytes(train_images_file.read(1), 'big') / 256

        label_value = int.from_bytes(train_labels_file.read(1), 'big')
        label = np.zeros((10, 1))
        label[label_value, 0] = 1

        train_set.append((image, label))
    return train_set


# plot and show the image
def show_image(img):
    image_ = img.reshape((28, 28))
    plt.imshow(image_, 'gray')
    plt.show()


# plot the cost function in Epochs
def plot_cost(cost):
    plt.style.use('grayscale')
    plt.xlabel('Epoch')
    plt.ylabel('average_cost')
    plt.title('Cost-Epoch diagram')
    plt.plot(cost, color='red')
    plt.show()


# shift images 4 pixels to the right and,
# make the left pixels black
def shift_images():
    IMAGE_LENGTH = 28
    FIRST_LAYER = 784
    output = []
    test_images = read_test_set()
    for i in range(len(test_images)):
        working_image = test_images[i]
        image_matrix = working_image[0].reshape((IMAGE_LENGTH, IMAGE_LENGTH))
        new_matrix = np.delete(image_matrix, IMAGE_LENGTH - 1, 1)
        new_matrix = np.delete(new_matrix, IMAGE_LENGTH - 2, 1)
        new_matrix = np.delete(new_matrix, IMAGE_LENGTH - 3, 1)
        new_matrix = np.delete(new_matrix, IMAGE_LENGTH - 4, 1)

        new_column = np.zeros(IMAGE_LENGTH)
        new_matrix = np.insert(new_matrix, 0, new_column, 1)
        new_matrix = np.insert(new_matrix, 0, new_column, 1)
        new_matrix = np.insert(new_matrix, 0, new_column, 1)
        new_matrix = np.insert(new_matrix, 0, new_column, 1)

        new_matrix = new_matrix.reshape(FIRST_LAYER, 1)
        output.append((new_matrix, working_image[1]))
    return output


# build mini-batches from the whole dataset
def build_mini_batch(dataset, bach_size, bach_count):
    tmp = []
    mini_batch = []
    for i in range(bach_count):
        tmp.clear()
        for j in range(bach_size):
            tmp.append(dataset[bach_size * i + j])
        mini_batch.append(tmp.copy())
    return mini_batch


# shuffle the input mini-bach
# using it on top of every Epoch iteration
def shuffle_mini_batch(dataset, bach_size, bach_count):
    random.shuffle(dataset)
    return build_mini_batch(dataset, bach_size, bach_count)
