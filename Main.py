import csv
import random
import time

import numpy as np
from activation_function import array_relu, array_relu_derivative, array_sigmoid, array_sigmoid_derivative, array_tanh, array_tanh_derivative
from generator_functions import generate_random_w, generate_random_b
from image_functions import read_training_set
from learning_functions import feed_forward_calculator, backpropagation, vectorized_backpropagation,\
    shifted_images_testing_backpropagation

ACTIVATION_FUNCTION = array_sigmoid
ACTIVATION_FUNCTION_DERIVATIVE = array_sigmoid_derivative
IMAGE_LENGTH = 28
FIRST_LAYER = 784
HIDDEN_LAYER = 16
LAST_LAYER = 10
LEARNING_RATE = 1
EPOCHS = 20
BATCH_SIZE = 5
BATCH_COUNT = 20


# calculate output, recognized and actual output for 100 training set data,
# and save it into a csv file named as report_1,
# and finally print the accuracy on the terminal
def report_1():
    wrong_recognition = 0
    right_recognition = 0
    training_set = read_training_set()
    with open('report_1.csv', mode='w') as employee_file:
        employee_writer = csv.writer(employee_file)

        employee_writer.writerow(['output of neural network', 'recognized number', 'actual number'])
        employee_writer.writerow([])
        for i in range(100):
            output = feed_forward_calculator(training_set[i][0], generate_random_w(), generate_random_b(), ACTIVATION_FUNCTION)[1][2]
            label = list(training_set[i][1]).index(max(training_set[i][1]))
            recognized_number = list(output).index(max(output))
            employee_writer.writerow([output, recognized_number, label])
            if recognized_number == label:
                right_recognition += 1
            else:
                wrong_recognition += 1
        print(f'Accuracy: {right_recognition / (right_recognition + wrong_recognition) * 100} %')
        return right_recognition / (right_recognition + wrong_recognition) * 100


# calculate the output of network with 100 training set data,
# and calculate the accuracy and the cost function for each Epoch
def report_2():
    global BATCH_SIZE, BATCH_COUNT, EPOCHS, LEARNING_RATE
    LEARNING_RATE = 1
    EPOCHS = 20
    BATCH_SIZE = 10
    BATCH_COUNT = 10
    backpropagation(BATCH_COUNT, BATCH_SIZE, EPOCHS, FIRST_LAYER, HIDDEN_LAYER, LAST_LAYER, LEARNING_RATE)


# calculate the output of network with 100 training set data,
# and calculate the accuracy and the cost function for each Epoch
# with vectorized backpropagation in 200 Epochs
def report_3():
    global BATCH_SIZE, BATCH_COUNT, EPOCHS, LEARNING_RATE
    LEARNING_RATE = 1
    EPOCHS = 200
    BATCH_SIZE = 5
    BATCH_COUNT = 20
    vectorized_backpropagation('train', BATCH_SIZE, BATCH_COUNT, EPOCHS, LEARNING_RATE, ACTIVATION_FUNCTION, ACTIVATION_FUNCTION_DERIVATIVE)


# calculate the final output, by training the whole data in 5 Epochs,
# and another Epoch for calculating the accuracy.
# it has a input that shows we are working with test/train data set.
def report_4(mode):
    global BATCH_SIZE, BATCH_COUNT, EPOCHS, LEARNING_RATE
    LEARNING_RATE = 1
    EPOCHS = 6
    if mode == 'train':
        BATCH_SIZE = 5
        BATCH_COUNT = 12000
    elif mode == 'test':
        BATCH_SIZE = 5
        BATCH_COUNT = 2000

    vectorized_backpropagation(mode, BATCH_SIZE, BATCH_COUNT, EPOCHS, LEARNING_RATE, ACTIVATION_FUNCTION, ACTIVATION_FUNCTION_DERIVATIVE)


# train the network with normal images and then test it with shifted images.
def bonus_report_1():
    global BATCH_SIZE, BATCH_COUNT, EPOCHS, LEARNING_RATE
    LEARNING_RATE = 1
    EPOCHS = 10
    BATCH_SIZE = 5
    BATCH_COUNT = 1000
    shifted_images_testing_backpropagation(BATCH_SIZE, BATCH_COUNT, EPOCHS, LEARNING_RATE, ACTIVATION_FUNCTION, ACTIVATION_FUNCTION_DERIVATIVE)


# using Relu function as another activation function,
# then calculate the accuracy and cost for each Epoch
def bonus_report_2():
    global BATCH_SIZE, BATCH_COUNT, EPOCHS, LEARNING_RATE, ACTIVATION_FUNCTION, ACTIVATION_FUNCTION_DERIVATIVE
    ACTIVATION_FUNCTION = array_relu
    ACTIVATION_FUNCTION_DERIVATIVE = array_relu_derivative
    LEARNING_RATE = 1.2
    EPOCHS = 50
    BATCH_SIZE = 5
    BATCH_COUNT = 50
    vectorized_backpropagation('test', BATCH_SIZE, BATCH_COUNT, EPOCHS, LEARNING_RATE, ACTIVATION_FUNCTION, ACTIVATION_FUNCTION_DERIVATIVE)


if __name__ == '__main__':
    starting_time = time.time()
    # report_1()
    # report_2()
    # report_3()
    # report_4('train')
    # report_4('test')
    # bonus_report_1()
    # bonus_report_2()
    end_time = time.time()
    print(f'Execution time is: {"{:0.4}".format(end_time - starting_time)} seconds.')