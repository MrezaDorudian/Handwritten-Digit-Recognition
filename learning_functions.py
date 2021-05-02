import numpy as np
from generator_functions import generate_random_w, generate_random_b, generate_zero_grad_a, generate_zero_grad_b,\
    generate_zero_grad_w
from image_functions import read_training_set, read_test_set, shift_images, plot_cost, shuffle_mini_batch
from activation_function import array_sigmoid, sigmoid_derivative


# calculate cost for an specific image
def calculate_cost(desired, actual):
    cost = 0
    for i in range(len(desired)):
        cost += np.square(actual[i, 0] - desired[i, 0])
    return cost


# calculate the output of neural network
# as feed forward without any learning
def feed_forward_calculator(image, w_matrix, b_vector, activation_function):
    w_matrix_layer_1, w_matrix_layer_2, w_matrix_layer_3 = w_matrix
    b_vector_layer_1, b_vector_layer_2, b_vector_layer_3 = b_vector

    z_vector_layer_1 = np.add(np.matmul(w_matrix_layer_1.copy(), image.copy()), b_vector_layer_1.copy())
    output_vector_layer_1 = activation_function(z_vector_layer_1.copy())
    # layer 1 output ready

    z_vector_layer_2 = np.add(np.matmul(w_matrix_layer_2.copy(), output_vector_layer_1.copy()), b_vector_layer_2.copy())
    output_vector_layer_2 = activation_function(z_vector_layer_2.copy())
    # layer 2 output ready

    z_vector_layer_3 = np.add(np.matmul(w_matrix_layer_3.copy(), output_vector_layer_2.copy()), b_vector_layer_3.copy())
    output_vector_layer_3 = activation_function(z_vector_layer_3.copy())
    # layer 3 output ready

    w_matrix = [w_matrix_layer_1, w_matrix_layer_2, w_matrix_layer_3]
    b_vector = [b_vector_layer_1, b_vector_layer_2, b_vector_layer_3]
    z_vector = [z_vector_layer_1, z_vector_layer_2, z_vector_layer_3]
    output_vector = [output_vector_layer_1, output_vector_layer_2, output_vector_layer_3]

    return w_matrix, output_vector, b_vector, z_vector


# calculate the backpropagation, and in fact it's the learning phase using 'for',
# for calculating the grads and updating them
def backpropagation(batch_size, batch_count, epochs, first_layer, hidden_layer, last_layer, learning_rate):
    training_set = read_test_set()[0:batch_count * batch_size]
    average_cost = []
    input_w_matrix = generate_random_w()
    input_b_vector = generate_random_b()

    for epoch in range(epochs):
        right_recognition = 0
        wrong_recognition = 0
        mini_batch = shuffle_mini_batch(training_set, batch_size, batch_count)
        cost = 0
        for batch_index in range(len(mini_batch)):
            grad_w_matrix_layer_1, grad_w_matrix_layer_2, grad_w_matrix_layer_3 = generate_zero_grad_w()
            grad_b_vector_layer_1, grad_b_vector_layer_2, grad_b_vector_layer_3 = generate_zero_grad_b()
            grad_a_vector_layer_1, grad_a_vector_layer_2, grad_a_vector_layer_3 = generate_zero_grad_a()

            for image in mini_batch[batch_index]:
                w_matrix, output_vector, b_vector, z_vector = feed_forward_calculator(image[0], input_w_matrix,
                                                                                      input_b_vector, array_sigmoid)

                for j in range(last_layer):
                    for k in range(hidden_layer):
                        grad_w_matrix_layer_3[j, k] += output_vector[1][k, 0] * sigmoid_derivative(z_vector[2][j, 0]) * (2 * output_vector[2][j, 0] - 2 * image[1][j, 0])
                for j in range(last_layer):
                    grad_b_vector_layer_3[j, 0] += sigmoid_derivative(z_vector[2][j, 0]) * 2 * (output_vector[2][j, 0] - image[1][j, 0])

                for k in range(hidden_layer):
                    for j in range(last_layer):
                        grad_a_vector_layer_2[k, 0] += w_matrix[2][j, k] * sigmoid_derivative(z_vector[2][j, 0]) * 2 * (output_vector[2][j, 0] - image[1][j, 0])

                for j in range(hidden_layer):
                    for k in range(hidden_layer):
                        grad_w_matrix_layer_2[j, k] += output_vector[0][k, 0] * sigmoid_derivative(z_vector[1][j, 0]) * grad_a_vector_layer_2[j, 0]

                for j in range(hidden_layer):
                    grad_b_vector_layer_2[j, 0] += sigmoid_derivative(z_vector[1][j, 0]) * grad_a_vector_layer_2[j, 0]

                for k in range(hidden_layer):
                    for j in range(hidden_layer):
                        grad_a_vector_layer_1[k, 0] += w_matrix[1][j, k] * sigmoid_derivative(z_vector[1][j, 0]) * grad_a_vector_layer_2[j, 0]

                for j in range(hidden_layer):
                    for k in range(first_layer):
                        grad_w_matrix_layer_1[j, k] += image[0][k] * sigmoid_derivative(z_vector[0][j, 0]) * grad_a_vector_layer_1[j, 0]

                for j in range(hidden_layer):
                    grad_b_vector_layer_1[j, 0] += sigmoid_derivative(z_vector[0][j, 0]) * grad_a_vector_layer_1[j, 0]

                label = list(image[1]).index(max(image[1]))
                recognized_number = list(output_vector[2]).index(max(output_vector[2]))
                if label == recognized_number:
                    right_recognition += 1
                else:
                    wrong_recognition += 1
                cost += calculate_cost(output_vector[2], image[1])

            w_matrix[0] -= (learning_rate * grad_w_matrix_layer_1 / batch_size)
            w_matrix[1] -= (learning_rate * grad_w_matrix_layer_2 / batch_size)
            w_matrix[2] -= (learning_rate * grad_w_matrix_layer_3 / batch_size)

            b_vector[0] -= (learning_rate * grad_b_vector_layer_1 / batch_size)
            b_vector[1] -= (learning_rate * grad_b_vector_layer_2 / batch_size)
            b_vector[2] -= (learning_rate * grad_b_vector_layer_3 / batch_size)

            input_w_matrix = (w_matrix[0], w_matrix[1], w_matrix[2])
            input_b_vector = (b_vector[0], b_vector[1], b_vector[2])
        accuracy = right_recognition / (right_recognition + wrong_recognition)
        average_cost.append(cost / (batch_size * batch_count))
        print(f'epoch {epoch} finished.')
        print(f'Accuracy: {accuracy * 100} %')
    print()
    print(f'Accuracy: {accuracy * 100} %')
    print(f'Number of Epochs: {epochs}')
    plot_cost(average_cost)


# calculate the backpropagation, and in fact it's the learning phase using
# matrix and vector functions in numpy, for calculating the grads and updating them
# it has a huge difference in learning time and reduces it a lot.
def vectorized_backpropagation(mode, batch_size, batch_count, epochs, learning_rate, activation_function,
                               activation_function_derivative):
    if mode == 'train':
        training_set = read_training_set()[0:batch_count * batch_size]
    elif mode == 'test':
        training_set = read_test_set()[0:batch_count * batch_size]
    elif mode == 'shifted':
        training_set = shift_images()[0:batch_count * batch_size]
    average_cost = []

    input_w_matrix = generate_random_w()
    input_b_vector = generate_random_b()
    for epoch in range(epochs):
        mini_batch = shuffle_mini_batch(training_set, batch_size, batch_count)
        right_recognition = 0
        wrong_recognition = 0
        cost = 0
        for batch_index in range(batch_count):
            grad_w_matrix_layer_1, grad_w_matrix_layer_2, grad_w_matrix_layer_3 = generate_zero_grad_w()
            grad_b_vector_layer_1, grad_b_vector_layer_2, grad_b_vector_layer_3 = generate_zero_grad_b()
            grad_a_vector_layer_1, grad_a_vector_layer_2, grad_a_vector_layer_3 = generate_zero_grad_a()

            for image in mini_batch[batch_index]:
                w_matrix, output_vector, b_vector, z_vector = feed_forward_calculator(image[0], input_w_matrix, input_b_vector, activation_function)

                label = list(image[1]).index(max(image[1]))
                recognized_number = list(output_vector[2]).index(max(output_vector[2]))
                if label == recognized_number:
                    right_recognition += 1
                else:
                    wrong_recognition += 1

                cost += calculate_cost(output_vector[2], image[1])

                grad_w_matrix_layer_3 += (activation_function_derivative(z_vector[2].copy()) * (2 * output_vector[2].copy() - 2 * image[1].copy())) @ output_vector[1].copy().transpose()
                grad_b_vector_layer_3 += activation_function_derivative(z_vector[2].copy()) * (2 * output_vector[2].copy() - 2 * image[1].copy())

                grad_a_vector_layer_2 += w_matrix[2].transpose() @ (activation_function_derivative(z_vector[2].copy()) * (2 * output_vector[2].copy() - 2 * image[1].copy()))
                grad_w_matrix_layer_2 += (activation_function_derivative(z_vector[1].copy()) * grad_a_vector_layer_2.copy()) @ output_vector[0].copy().transpose()
                grad_b_vector_layer_2 += activation_function_derivative(z_vector[1].copy()) * grad_a_vector_layer_2.copy()

                grad_a_vector_layer_1 += w_matrix[1].transpose() @ (activation_function_derivative(z_vector[1].copy()) * grad_a_vector_layer_2.copy())
                grad_w_matrix_layer_1 += (activation_function_derivative(z_vector[0].copy()) * grad_a_vector_layer_1.copy()) @ image[0].transpose()
                grad_b_vector_layer_1 += activation_function_derivative(z_vector[0].copy()) * grad_a_vector_layer_1.copy()

            w_matrix[0] -= learning_rate * grad_w_matrix_layer_1 / batch_size
            w_matrix[1] -= learning_rate * grad_w_matrix_layer_2 / batch_size
            w_matrix[2] -= learning_rate * grad_w_matrix_layer_3 / batch_size

            b_vector[0] -= learning_rate * grad_b_vector_layer_1 / batch_size
            b_vector[1] -= learning_rate * grad_b_vector_layer_2 / batch_size
            b_vector[2] -= learning_rate * grad_b_vector_layer_3 / batch_size

            input_w_matrix = (w_matrix[0], w_matrix[1], w_matrix[2])
            input_b_vector = (b_vector[0], b_vector[1], b_vector[2])

        accuracy = right_recognition / (right_recognition + wrong_recognition)
        average_cost.append(cost / (batch_size * batch_count))
        print(f'epoch {epoch} finished.')
        print(f'Accuracy: {accuracy * 100} %')
    print()
    print(f'Accuracy: {accuracy * 100} %')
    print(f'Number of Epochs: {epochs}')
    plot_cost(average_cost)


# train the network with normal pictures and then,
# test it with shifted images with vectorized backpropagation
# and calculate the accuracy and the cost function
def shifted_images_testing_backpropagation(batch_size, batch_count, epochs, learning_rate, activation_function,
                                           activation_function_derivative):
    training_set = read_test_set()[0:batch_count * batch_size]

    test_set = shift_images()[0:batch_count * batch_size]

    input_w_matrix = generate_random_w()
    input_b_vector = generate_random_b()
    for epoch in range(epochs):
        mini_batch = shuffle_mini_batch(training_set, batch_size, batch_count)
        for batch_index in range(batch_count):
            grad_w_matrix_layer_1, grad_w_matrix_layer_2, grad_w_matrix_layer_3 = generate_zero_grad_w()
            grad_b_vector_layer_1, grad_b_vector_layer_2, grad_b_vector_layer_3 = generate_zero_grad_b()
            grad_a_vector_layer_1, grad_a_vector_layer_2, grad_a_vector_layer_3 = generate_zero_grad_a()

            for image in mini_batch[batch_index]:
                w_matrix, output_vector, b_vector, z_vector = feed_forward_calculator(image[0], input_w_matrix, input_b_vector, activation_function)

                grad_w_matrix_layer_3 += (activation_function_derivative(z_vector[2].copy()) * (2 * output_vector[2].copy() - 2 * image[1].copy())) @ output_vector[1].copy().transpose()
                grad_b_vector_layer_3 += activation_function_derivative(z_vector[2].copy()) * (2 * output_vector[2].copy() - 2 * image[1].copy())

                grad_a_vector_layer_2 += w_matrix[2].transpose() @ (activation_function_derivative(z_vector[2].copy()) * (2 * output_vector[2].copy() - 2 * image[1].copy()))
                grad_w_matrix_layer_2 += (activation_function_derivative(z_vector[1].copy()) * grad_a_vector_layer_2.copy()) @ output_vector[0].copy().transpose()
                grad_b_vector_layer_2 += activation_function_derivative(z_vector[1].copy()) * grad_a_vector_layer_2.copy()

                grad_a_vector_layer_1 += w_matrix[1].transpose() @ (activation_function_derivative(z_vector[1].copy()) * grad_a_vector_layer_2.copy())
                grad_w_matrix_layer_1 += (activation_function_derivative(z_vector[0].copy()) * grad_a_vector_layer_1.copy()) @ image[0].transpose()
                grad_b_vector_layer_1 += activation_function_derivative(z_vector[0].copy()) * grad_a_vector_layer_1.copy()

            w_matrix[0] -= learning_rate * grad_w_matrix_layer_1 / batch_size
            w_matrix[1] -= learning_rate * grad_w_matrix_layer_2 / batch_size
            w_matrix[2] -= learning_rate * grad_w_matrix_layer_3 / batch_size

            b_vector[0] -= learning_rate * grad_b_vector_layer_1 / batch_size
            b_vector[1] -= learning_rate * grad_b_vector_layer_2 / batch_size
            b_vector[2] -= learning_rate * grad_b_vector_layer_3 / batch_size

            input_w_matrix = (w_matrix[0], w_matrix[1], w_matrix[2])
            input_b_vector = (b_vector[0], b_vector[1], b_vector[2])

    right_recognition = 0
    wrong_recognition = 0
    for img in test_set:
        output = feed_forward_calculator(img[0], input_w_matrix, input_b_vector, activation_function)[1][2]
        label = list(img[1]).index(max(img[1]))
        recognized_number = list(output).index(max(output))
        if label == recognized_number:
            right_recognition += 1
        else:
            wrong_recognition += 1

    accuracy = right_recognition / (right_recognition + wrong_recognition)

    print(f'Accuracy for shifted images after learning with original images: {accuracy * 100} %')