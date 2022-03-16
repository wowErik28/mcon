from mcon.node.base_node import Node

'''
The BaseFunction can be considered as the model that do not need to
realize backward. The Function object must realize all.
'''
class BaseFunction(object):

    def __init__(self):

        self.is_training = True

    def forward(self, *args):
        raise NotImplementedError

    def train(self):
        self.is_training = True

    def eval(self):
        self.is_training = False

    def parameter(self):
        '''
        Find all the parameters of this function
        :return: list(Parameter)
        '''
        parameter_list = []
        for value in list(self.__dict__.values()):

            if isinstance(value, Node):
                if value.is_parameter:
                    parameter_list.append(value)

            if isinstance(value, BaseFunction):
                parameter_list.extend(value.parameter())

        return parameter_list

    def freeze(self):
        #You also needs to call model(x).is_trainable = False after calling freeze
        for node in self.parameter():
            node.is_trainable = False

    def __call__(self, *args):

        return self.forward(*args)

class Function(BaseFunction):

    def __init__(self):

        super().__init__()
        #Use in forward. There maybe some variables that needs to save for the backward.
        self.save_for_backward = {}

        #The sequence of this list corresponds with the results of backward
        self.grad_node_list = []

    def _append_weights(self):
        #Append parameters of this function to grad_node_list if necessary.
        pass

    def process_input_nodes(self, *args):

        #init the grad_node_list
        del self.grad_node_list
        self.grad_node_list = []
        self.grad_node_list.extend(args)

        #process the nodes
        for node in args:
            node.grad_fn.append(self)

            #Whether the node is leaf.
            if node.output_fn is not None:
                node.require_grad = True

        #append weights
        self._append_weights()

    def process_output_node(self, *args):
        for node in args:
            node.output_fn = self

    def backward(self, grad_pre):
        '''
        :param grad_pre: np.ndarray
        :return: tuple(np.ndarray)
        '''
        raise NotImplementedError

    def __call__(self, *args):
        self.process_input_nodes(*args)

        return self.forward(*args)