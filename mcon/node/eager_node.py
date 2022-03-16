import numpy as np

from mcon.node.base_node import Node
from mcon.function.functions import Add, Multiply, Minus, \
    Divide, Pow, Reshape, Dot, Exp, ReduceMean, ReduceSum

class EagerNode(Node):

    @classmethod
    def clone_from_node(cls, node):
        eager_node =  EagerNode(node.data, require_grad = node.require_grad,
                         is_trainable=node.is_trainable, is_parameter=node.is_parameter)
        # np.ndarray. self.grad.shape == self.data.shape
        eager_node.grad = node.grad

        # Function The functions which take this node as output
        eager_node.output_fn = node.output_fn

        # list(Function) The functions that should provides this node with gradients
        eager_node.grad_fn = node.grad_fn

        # Use in backward
        eager_node.grad_count = node.grad_count
        return eager_node
    def decorate_input_to_eagernode(self, other):
        if isinstance(other, Node):
            return other

        return EagerNode(np.array(other), require_grad=False, is_parameter=False)

    def __add__(self, other):
        other = self.decorate_input_to_eagernode(other)
        output_node = Add()(self, other)
        return EagerNode.clone_from_node(output_node)

    def __mul__(self, other):
        other = self.decorate_input_to_eagernode(other)
        output_node = Multiply()(self, other)
        return EagerNode.clone_from_node(output_node)

    def __sub__(self, other):
        other = self.decorate_input_to_eagernode(other)
        output_node = Minus()(self, other)
        return EagerNode.clone_from_node(output_node)

    def __truediv__(self, other):
        other = self.decorate_input_to_eagernode(other)
        output_node = Divide()(self, other)
        return EagerNode.clone_from_node(output_node)

    def __pow__(self, power):
        power = self.decorate_input_to_eagernode(power)
        output_node = Pow()(self, power)
        return EagerNode.clone_from_node(output_node)

    def reshape(self, shape):
        output_node = Reshape()(self, shape)
        return EagerNode.clone_from_node(output_node)

    def __matmul__(self, other):
        other = self.decorate_input_to_eagernode(other)
        output_node = Dot()(self, other)
        return EagerNode.clone_from_node(output_node)

    def exp(self):
        output_node = Exp()(self)
        return EagerNode.clone_from_node(output_node)

    def reduce_mean(self, axis=-1, keepdims=False):
        output_node = ReduceMean()(self, axis, keepdims)
        return EagerNode.clone_from_node(output_node)

    def reduce_sum(self, axis=-1, keepdims=False):
        output_node = ReduceSum()(self, axis, keepdims)
        return EagerNode.clone_from_node(output_node)