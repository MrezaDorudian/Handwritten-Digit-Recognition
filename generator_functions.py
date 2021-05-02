import numpy as np

FIRST_LAYER = 784
HIDDEN_LAYER = 16
LAST_LAYER = 10


# generating random W for the input weights
def generate_random_w():
    w_matrix_layer_1 = np.random.normal(size=(HIDDEN_LAYER, FIRST_LAYER))
    w_matrix_layer_2 = np.random.normal(size=(HIDDEN_LAYER, HIDDEN_LAYER))
    w_matrix_layer_3 = np.random.normal(size=(LAST_LAYER, HIDDEN_LAYER))
    return w_matrix_layer_1, w_matrix_layer_2, w_matrix_layer_3


# generating random b for the input biases
def generate_random_b():
    b_vector_layer_1 = np.zeros((HIDDEN_LAYER, 1))
    b_vector_layer_2 = np.zeros((HIDDEN_LAYER, 1))
    b_vector_layer_3 = np.zeros((LAST_LAYER, 1))
    return b_vector_layer_1, b_vector_layer_2, b_vector_layer_3


# generating a zero matrix based on W dimensions in each layer
def generate_zero_grad_w():
    grad_w_matrix_layer_1 = np.zeros((HIDDEN_LAYER, FIRST_LAYER))
    grad_w_matrix_layer_2 = np.zeros((HIDDEN_LAYER, HIDDEN_LAYER))
    grad_w_matrix_layer_3 = np.zeros((LAST_LAYER, HIDDEN_LAYER))
    return grad_w_matrix_layer_1, grad_w_matrix_layer_2, grad_w_matrix_layer_3


# generating a zero vector based on b dimension in each layer
def generate_zero_grad_b():
    grad_b_vector_layer_1 = np.zeros((HIDDEN_LAYER, 1))
    grad_b_vector_layer_2 = np.zeros((HIDDEN_LAYER, 1))
    grad_b_vector_layer_3 = np.zeros((LAST_LAYER, 1))
    return grad_b_vector_layer_1, grad_b_vector_layer_2, grad_b_vector_layer_3


# generating a zero vector based on a dimension in each layer
def generate_zero_grad_a():
    grad_a_vector_layer_1 = np.zeros((HIDDEN_LAYER, 1))
    grad_a_vector_layer_2 = np.zeros((HIDDEN_LAYER, 1))
    grad_a_vector_layer_3 = np.zeros((LAST_LAYER, 1))
    return grad_a_vector_layer_1, grad_a_vector_layer_2, grad_a_vector_layer_3
