import numpy as np

class Node(object):

    def __init__(self, data, require_grad = False, is_trainable=True, is_parameter=False):
        '''
        :param data: np.ndarray
        :param require_grad: bool. Whether this node needs to call backward.
        :param is_trainable: bool. Whether this node needs to call backward and update self value
        :param is_parameter: bool. Whether use optimizer to update the data of this node
        '''
        self.data = data
        self.require_grad = require_grad
        self.is_trainable = is_trainable
        self.is_parameter = is_parameter

        #np.ndarray. self.grad.shape == self.data.shape
        self.grad = None

        #Function The functions which take this node as output
        self.output_fn = None

        # list(Function) The functions that should provides this node with gradients
        self.grad_fn = []

        #Use in backward
        self.grad_count = 0

    def shape(self):
        return self.data.shape

    def zero_grad(self):
        del self.grad
        self.grad = np.zeros_like(self.data)

    def _update_grad(self, grad):

        self.grad_count += 1
        if not self.require_grad or not self.is_trainable:
            return

        if self.grad is None:
            self.grad = grad + 0
        else:
            self.grad += grad

    def backward(self, grad_pre):
        '''
        We need to consider require_grad, is_trainable, grad_count.
        :param grad_pre: ndarray. Its shape is the same as the that of output of forward.
        '''
        if not self.is_trainable:
            return

        if not self.require_grad:
            return

        if self.grad_count != len(self.grad_fn):
            return

        self.grad_count = 0
        if self.output_fn is not None:
            grads = self.output_fn.backward(grad_pre)

            if isinstance(grads, np.ndarray):
                self.output_fn.grad_node_list[0]._update_grad(grads)

            if isinstance(grads, tuple) or isinstance(grads, list):
                for i, grad in enumerate(grads):
                    self.output_fn.grad_node_list[i]._update_grad(grad)

            for node in self.output_fn.grad_node_list:
                node.backward(node.grad)