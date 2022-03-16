import numpy as np
import pickle

def broadcast_backward(grad, self_shape):
    #make grad.shape == self_shape
    while len(grad.shape) > len(self_shape):
        grad = np.sum(grad, axis=0)

    for i in range(len(grad.shape)):
        if grad.shape[i] != self_shape[i]:
            grad = np.sum(grad, axis=i, keepdims=True)

    return grad

def save_model(model, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(model, file)

def load_model(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)

def dense_to_onehot(labels_dense, num_class):
    '''
    :param labels_dense: (N,)
    :param num_class: int
    :return: (N, num_class)
    '''
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_class
    labels_onehot = np.zeros((num_labels, num_class))
    labels_onehot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_onehot