import numpy as np

from mcon.function.base_function import *
from mcon.utils.utils import broadcast_backward

'''
No parameter Function, and these functions are involved in 
operator overload of EagerNode
'''

class Add(Function):

    def forward(self, *args):
        '''
        :param a: Node
        :param b: Node
        :return: Node
        '''

        datas = [node.data for node in args]
        self.save_for_backward['datas'] = datas
        result = sum(datas)
        output = Node(result, require_grad=True, is_trainable=True)
        self.process_output_node(output)
        return output

    def backward(self, grad_pre):

        datas = self.save_for_backward['datas']

        return [broadcast_backward(grad_pre, data.shape) for data in datas]

class Minus(Function):

    def forward(self, a, b):
        a = a.data
        b = b.data

        result = a - b
        self.save_for_backward['a'] = a
        self.save_for_backward['b'] = b

        output = Node(result, require_grad=True, is_trainable=True)
        self.process_output_node(output)

        return output

    def backward(self, grad_pre):
        a = self.save_for_backward['a']
        b = self.save_for_backward['b']

        return broadcast_backward(grad_pre, a.shape), \
               broadcast_backward(-grad_pre, b.shape)

class Multiply(Function):

    def forward(self, a, b):
        a = a.data
        b = b.data

        result = a * b
        self.save_for_backward['a'] = a
        self.save_for_backward['b'] = b

        output = Node(result, require_grad=True, is_trainable=True)
        self.process_output_node(output)

        return output

    def backward(self, grad_pre):
        a = self.save_for_backward['a']
        b = self.save_for_backward['b']

        return broadcast_backward(grad_pre * b, a.shape), broadcast_backward(grad_pre * a, b.shape)

class Divide(Function):

    def forward(self, a, b):
        a = a.data
        b = b.data

        result = a / b
        self.save_for_backward['a'] = a
        self.save_for_backward['b'] = b

        output = Node(result, require_grad=True, is_trainable=True)
        self.process_output_node(output)

        return output

    def backward(self, grad_pre):
        a = self.save_for_backward['a']
        b = self.save_for_backward['b']

        return broadcast_backward(grad_pre * (1 / b), a.shape), \
               broadcast_backward(-grad_pre * (a / b**2), b.shape)

class Pow(Function):

    def forward(self, a, b):
        '''
        :param a: Node
        :param b: Node. shape = (1,)
        :return: Node
        '''
        a = a.data
        b = b.data

        result = a ** b
        self.save_for_backward['a'] = a
        self.save_for_backward['b'] = b
        self.save_for_backward['result'] = result

        output = Node(result, require_grad=True, is_trainable=True)
        self.process_output_node(output)

        return output

    def backward(self, grad_pre):
        a = self.save_for_backward['a']
        b = self.save_for_backward['b']
        result = self.save_for_backward['result']
        if b == 0:
            return np.zeros_like(a), np.sum(np.log(a) * result)

        return grad_pre * result * b / a, \
               np.sum(grad_pre * np.log(a) * result)

class Reshape(Function):

    def __call__(self, x, shape):
        self.process_input_nodes(x)
        return self.forward(x, shape)

    def forward(self, x, shape):
        x = x.data
        self.save_for_backward['old_shape'] = x.shape

        result = np.reshape(x, shape)
        output = Node(result, require_grad=True, is_trainable=True)
        self.process_output_node(output)
        return output


    def backward(self, grad_pre):
        old_shape = self.save_for_backward['old_shape']

        return grad_pre.reshape(old_shape)

class Dot(Function):

    def forward(self, a, b):
        a = a.data
        b = b.data

        result = a @ b
        self.save_for_backward['a'] = a
        self.save_for_backward['b'] = b

        output = Node(result, require_grad=True, is_trainable=True)
        self.process_output_node(output)

        return output

    def backward(self, grad_pre):
        a = self.save_for_backward['a']
        b = self.save_for_backward['b']

        return grad_pre @ b.T, a.T @ grad_pre

class Exp(Function):

    def forward(self, x):
        x = x.data

        result = np.exp(x)
        self.save_for_backward['result'] = result

        output = Node(result, require_grad=True, is_trainable=True)
        self.process_output_node(output)
        return output


    def backward(self, grad_pre):
        result = self.save_for_backward['result']

        return grad_pre * result

class ReduceSum(Function):
    def __call__(self, x, axis=-1, keepdims=False):
        self.process_input_nodes(x)
        return self.forward(x, axis, keepdims)

    def forward(self, x, axis, keepdims):
        x = x.data

        result = np.sum(x, axis=axis, keepdims=keepdims)
        self.save_for_backward['axis'] = axis
        self.save_for_backward['keepdims'] = keepdims
        self.save_for_backward['repeats'] = x.shape[axis]

        output = Node(result, require_grad=True, is_trainable=True)
        self.process_output_node(output)
        return output


    def backward(self, grad_pre):
        axis = self.save_for_backward['axis']
        keepdims = self.save_for_backward['keepdims']
        repeats = self.save_for_backward['repeats']
        if not keepdims:
            grad_pre = np.expand_dims(grad_pre, axis=axis)

        return grad_pre.repeat(repeats, axis=axis)


class ReduceMean(Function):

    def __call__(self, x, axis=-1, keepdims=False):
        self.process_input_nodes(x)
        return self.forward(x, axis, keepdims)

    def forward(self, x, axis, keepdims):
        x = x.data

        result = np.mean(x, axis=axis, keepdims=keepdims)
        self.save_for_backward['axis'] = axis
        self.save_for_backward['keepdims'] = keepdims
        self.save_for_backward['repeats'] = x.shape[axis]

        output = Node(result, require_grad=True, is_trainable=True)
        self.process_output_node(output)
        return output

    def backward(self, grad_pre):
        axis = self.save_for_backward['axis']
        keepdims = self.save_for_backward['keepdims']
        repeats = self.save_for_backward['repeats']
        if not keepdims:
            grad_pre = np.expand_dims(grad_pre, axis=axis)

        return grad_pre.repeat(repeats, axis=axis) * 1/repeats