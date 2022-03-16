import numpy as np

from mcon.function.base_function import BaseFunction, Function
from mcon.node.eager_node import EagerNode
from mcon.utils.utils import broadcast_backward, dense_to_onehot
from mcon.utils.init import xuniform

'''
These Functions are not involved the operator overload of EagerNode.
'''
class MSELoss(Function):

    def forward(self, input, target):
        '''
        :param input: (N,..)
        :param target: (N,..)
        :return:
        '''
        a = input.data
        b = target.data


        mid_result = a - b
        n = np.cumprod(mid_result.shape)[-1]

        result = np.mean(mid_result**2)
        self.save_for_backward['mid_result'] = mid_result
        self.save_for_backward['n'] = n
        self.save_for_backward['a'] = a
        self.save_for_backward['b'] = b

        output = EagerNode(result, require_grad=True, is_trainable=True)
        self.process_output_node(output)

        return output

    def backward(self, grad_pre=1.):
        mid_result = self.save_for_backward['mid_result']
        n = self.save_for_backward['n']
        a = self.save_for_backward['a']
        b = self.save_for_backward['b']

        grad_a = grad_pre * n /2 * mid_result
        grad_b = -grad_pre * n /2 * mid_result
        if grad_a.shape != a.shape:
            grad_a = broadcast_backward(grad_a, a.shape)
        if grad_b.shape != b.shape:
            grad_b = broadcast_backward(grad_b, b.shape)

        return grad_a, grad_b

class CrossEntropyLoss(Function):

    def __call__(self, input, target):
        self.process_input_nodes(input)

        return self.forward(input, target)

    def forward(self, input, target):
        '''
        :param input: Node. input.data.shape = (N,C)
        :param target: Node. target.data.shape = (N,) dtype = np.longlong
        :return: Node. data.shape = (1,)
        '''
        input = input.data
        target = target.data

        assert target.dtype == np.longlong

        N, C = input.shape
        target = dense_to_onehot(target, C) #(N, C)
        self.save_for_backward['bsz'] = N
        self.save_for_backward['input'] = input
        self.save_for_backward['target'] = target

        input = -np.log( np.sum(input * target, axis=-1) + 1e-6 ) #(N,)
        result = np.mean(input)
        output = EagerNode(result, require_grad=True, is_trainable=True)
        self.process_output_node(output)

        return output

    def backward(self, grad_pre=1.):
        bsz = self.save_for_backward['bsz']
        input = self.save_for_backward['input']
        target = self.save_for_backward['target']

        grad_input = -1 / (input + 1e-6) * input / bsz

        return grad_input

class ReLU(Function):

    def forward(self, x):
        x = 0 + x.data #get a new ndarray

        mask = np.where(x < 0)
        self.save_for_backward['mask'] = mask

        x[mask] = 0.
        result = x

        output = EagerNode(result, require_grad=True, is_trainable=True)
        self.process_output_node(output)

        return output

    def backward(self, grad_pre):
        mask = self.save_for_backward['mask']
        grad_pre[mask] = 0.

        return grad_pre

class Sigmoid(Function):

    def forward(self, x):
        x = x.data  # get a new ndarray

        result = 1 / (1 + np.exp(-x))
        self.save_for_backward['result'] = result

        output = EagerNode(result, require_grad=True, is_trainable=True)
        self.process_output_node(output)

        return output

    def backward(self, grad_pre):
        result = self.save_for_backward['result']

        return grad_pre * result * (1 - result)

class Softmax(BaseFunction):

    def forward(self, x):
        '''
        :param x: EagerNode. shape=(N, C)
        :return: EagerNode. shape=(N, C)
        '''
        shift = np.max(x.data, axis=-1, keepdims=True) #(N, 1)
        shift = EagerNode(shift, require_grad=False, is_parameter=False)

        exp_x = (x - shift).exp() #(N, C)
        exp_x_sum = exp_x.reduce_sum(axis=-1, keepdims=True) #(N, 1)

        return exp_x / exp_x_sum

class Linear(BaseFunction):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.init_parameters(in_features, out_features, bias, dtype=np.float)
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
    def init_parameters(self, in_features, out_features, bias=True, dtype=np.float):

        k = np.sqrt(in_features)
        self.W = EagerNode(np.random.uniform(-k, k, size=(in_features, out_features)),
                           require_grad=True, is_trainable=True, is_parameter=True)
        if bias:
            self.b = EagerNode(np.random.uniform(-k, k, size=(out_features,)),
                               require_grad=True, is_trainable=True, is_parameter=True)
    def forward(self, x):
        return x @ self.W + self.b

class ConvFunction(Function):

    def get_outputL(self, in_L, kernel_size, padding, stride):
        output_L = (in_L - kernel_size + 2 * padding) / stride + 1
        return np.int(np.floor(output_L))

class Conv1DFunction(ConvFunction):

    def __call__(self, x, filter, bias):

        #bias may be None
        if self.has_bias:
            self.process_input_nodes(x, filter, bias)
        else:
            self.process_input_nodes(x, filter)

        return self.forward(x, filter, bias)

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, has_bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.has_bias = has_bias

    def forward(self, x, filter, bias):
        '''
        :param x: Node(N,C,L)
        :param filter: Node(out_channels, C, kernel_size)
        :param bias: Node(out_channels, ) or None
        :return: Node
        '''
        x = x.data
        filter = filter.data

        N, C, L = x.shape
        #padding input
        if self.padding != 0:
            after_padding = np.zeros( shape=(N, C, L + 2*self.padding),
                                      dtype=x.dtype)
            after_padding[:, :, self.padding : self.padding + L] = x
        else:
            after_padding = x

        self.save_for_backward['after_padding_3dims'] = after_padding
        #(N, out_channels, C, L + 2*self.padding)
        after_padding = np.expand_dims(after_padding, axis=1).repeat(self.out_channels, axis=1)

        #init output_template
        output_L = self.get_outputL(L, self.kernel_size,
                                    self.padding, self.stride)
        output_template = np.zeros(shape=(N, self.out_channels, output_L))

        #calculate
        for L_i in range(output_L):
            ptr = L_i * self.stride

            #(N, out_channels, C, kernel_size)
            mid_result = after_padding[:, :, :, ptr : ptr+self.kernel_size] * filter

            #(N, out_channels)
            result = np.sum(np.sum(mid_result, axis=-1, keepdims=False),
                            axis=-1, keepdims=False)
            if self.has_bias:
                result += bias.data

            output_template[:, :, L_i] = result

        output = EagerNode(output_template, require_grad=True, is_trainable=True)
        self.process_output_node(output)

        self.save_for_backward['filter'] = filter
        self.save_for_backward['L'] = L

        return output

    def backward(self, grad_pre):
        #grad_pre.shape = (N, out_channels, output_L)
        N, _, output_L = grad_pre.shape

        #(N, out_channels, in_channels, L + 2*self.padding)
        after_padding = np.expand_dims(self.save_for_backward['after_padding_3dims'],
                                       axis=1).repeat(self.out_channels, 1)

        #(out_channels, in_channels, kernel_size)
        filter = self.save_for_backward['filter'] + 0 #return a new filter

        # (out_channels, in_channels, kernel_size)
        grad_filter = np.zeros_like(filter)

        filter = np.expand_dims(filter, 0) #(1, out_channels, in_channels, kernel_size)
        grad_x = np.zeros_like(self.save_for_backward['after_padding_3dims'])

        for L_i in range(output_L):
            #(N, out_channels, in_channels, kernel_size)
            mid_result = grad_pre[:, :, L_i].reshape(N, self.out_channels, 1, 1) * filter
            #(N, in_channels, kernel_size)
            mid_result = np.sum(mid_result, axis=1)
            ptr = L_i*self.stride
            grad_x[:, :, ptr : ptr+self.kernel_size] += mid_result

            #(N, out_channels, in_channels, kernel_size)
            mid_result = grad_pre[:, :, L_i].reshape(N, self.out_channels, 1, 1) * \
                         after_padding[:, :, :, ptr : ptr+self.kernel_size]
            mid_result = np.sum(mid_result, axis=0)
            grad_filter += mid_result

        grad_x = grad_x[:, :, self.padding: self.padding + self.save_for_backward['L']]

        if self.has_bias:
            grad_bias = np.sum(np.sum(grad_pre, axis=0), axis=-1)
            return grad_x, grad_filter, grad_bias

        return grad_x, grad_filter

class Conv1D(BaseFunction):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, has_bias=True):
        super().__init__()
        self.init_parameters(in_channels, out_channels, kernel_size, has_bias)
        self.conv = Conv1DFunction(in_channels, out_channels, kernel_size, stride,
                 padding, has_bias)

    def init_parameters(self, in_channels, out_channels, kernel_size, has_bias):
        filter_k = 1/(in_channels * kernel_size)
        filter_shape = (out_channels, in_channels, kernel_size)
        self.filters = EagerNode(xuniform(filter_shape, filter_k), require_grad=True,
                                 is_trainable=True, is_parameter=True)

        if has_bias:
            bias_k = filter_k
            bias_shape = (out_channels, )
            self.bias = EagerNode(xuniform(bias_shape, bias_k), require_grad=True,
                                 is_trainable=True, is_parameter=True)
        else:
            self.bias = None

    def forward(self, x):
        return self.conv(x, self.filters, self.bias)

class Conv2DFunction(ConvFunction):

    def __call__(self, x, filter, bias):

        #bias may be None
        if self.has_bias:
            self.process_input_nodes(x, filter, bias)
        else:
            self.process_input_nodes(x, filter)

        return self.forward(x, filter, bias)

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, has_bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        ###
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple) or isinstance(kernel_size, list):
            self.kernel_size = kernel_size
        else:
            raise Exception('kernel size should be single int or tuple or list')

        ###
        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif isinstance(stride, tuple) or isinstance(stride, list):
            self.stride = stride
        else:
            raise Exception('stride should be single int or tuple or list')

        ###
        if isinstance(padding, int):
            self.padding = (padding, padding)
        elif isinstance(padding, tuple) or isinstance(padding, list):
            self.padding = padding
        else:
            raise Exception('padding should be single int or tuple or list')

        self.has_bias = has_bias

    def forward(self, x, filter, bias):
        '''
        :param x: Node(N,C,H,W)
        :param filter: Node(out_channels, C, kernel_size[0], kernel_size[1])
        :param bias: Node(out_channels, ) or None
        :return: Node
        '''
        x = x.data
        filter = filter.data

        N, C, H, W = x.shape

        #padding input
        if self.padding != 0:
            after_padding = np.zeros( shape=(N, C, H + 2*self.padding[0], W + 2*self.padding[1]),
                                      dtype=x.dtype)
            after_padding[:, :, self.padding[0] : self.padding[0] + H,
                          self.padding[1] : self.padding[1] + W] = x
        else:
            after_padding = x

        self.save_for_backward['after_padding_4dims'] = after_padding
        #(N, out_channels, C, H + 2*self.padding[0], W + 2*self.padding[1])
        after_padding = np.expand_dims(after_padding, axis=1).repeat(self.out_channels, axis=1)

        #init output_template
        output_H = self.get_outputL(H, self.kernel_size[0],
                                    self.padding[0], self.stride[0])
        output_W = self.get_outputL(W, self.kernel_size[1],
                                    self.padding[1], self.stride[1])

        output_template = np.zeros(shape=(N, self.out_channels, output_H, output_W))

        #calculate
        for H_i in range(output_H):
            H_ptr = H_i * self.stride[0]
            for W_i in range(output_W):
                W_ptr = W_i * self.stride[1]

                #(N, out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])
                mid_result = after_padding[:, :, :, H_ptr : H_ptr + self.kernel_size[0],
                              W_ptr : W_ptr + self.kernel_size[1]] * filter
                #(N, out_channels)
                result = np.sum(np.sum(np.sum(mid_result, axis=-1), axis=-1), axis=-1)
                if self.has_bias:
                    result += bias.data
                output_template[:, :, H_i, W_i] = result

        output = EagerNode(output_template, require_grad=True, is_trainable=True)
        self.process_output_node(output)

        self.save_for_backward['filter'] = filter
        self.save_for_backward['H, W'] = (H, W)

        return output

    def backward(self, grad_pre):
        #grad_pre.shape = (N, out_channels, output_H, output_W)
        N, _, output_H, output_W = grad_pre.shape

        #(N, out_channels, C, H + 2*self.padding[0], W + 2*self.padding[1])
        after_padding = np.expand_dims(self.save_for_backward['after_padding_4dims'],
                                       axis=1).repeat(self.out_channels, 1)

        #(out_channels, in_channels, kernel_size[0], kernel_size[1])
        filter = self.save_for_backward['filter'] + 0 #return a new filter

        # (out_channels, in_channels, kernel_size[0], kernel_size[1])
        grad_filter = np.zeros_like(filter)

        filter = np.expand_dims(filter, 0) #(1, out_channels, in_channels, kernel_size[0], kernel_size[1])
        # (N, C, H + 2*self.padding[0], W + 2*self.padding[1])
        grad_x = np.zeros_like(self.save_for_backward['after_padding_4dims'])

        for H_i in range(output_H):
            H_ptr = H_i * self.stride[0]
            for W_i in range(output_W):
                #(N, out_channels, in_channels, kernel_size[0], kernel_size[1])
                mid_result = grad_pre[:, :, H_i, W_i].reshape(N, self.out_channels, 1, 1, 1) * filter
                #(N, in_channels, kernel_size[0], kernel_size[1])
                mid_result = np.sum(mid_result, axis=1)
                W_ptr = W_i * self.stride[1]
                grad_x[:, :, H_ptr : H_ptr + self.kernel_size[0],
                       W_ptr: W_ptr + self.kernel_size[1]] += mid_result


                mid_result = grad_pre[:, :, H_i, W_i].reshape(N, self.out_channels, 1, 1, 1) * \
                             after_padding[:, :, :, H_ptr : H_ptr + self.kernel_size[0],
                                           W_ptr: W_ptr + self.kernel_size[1]]
                mid_result = np.sum(mid_result, axis=0)
                grad_filter += mid_result

        H, W = self.save_for_backward['H, W']
        grad_x = grad_x[:, :, self.padding[0]: self.padding[0] + H, self.padding[1]: self.padding[1] + W]

        if self.has_bias:
            grad_bias = np.sum(np.sum(np.sum(grad_pre, axis=0), axis=-1), axis=-1)
            return grad_x, grad_filter, grad_bias

        return grad_x, grad_filter

class Conv2D(BaseFunction):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, has_bias=True):
        super().__init__()

        self.conv = Conv2DFunction(in_channels, out_channels, kernel_size, stride,
                 padding, has_bias)
        self.init_parameters(self.conv.in_channels, self.conv.out_channels,
                             self.conv.kernel_size, self.conv.has_bias)

    def init_parameters(self, in_channels, out_channels, kernel_size, has_bias):
        filter_k = 1/(in_channels * kernel_size[0] * kernel_size[1])
        filter_shape = (out_channels, in_channels, kernel_size[0], kernel_size[1])
        self.filters = EagerNode(xuniform(filter_shape, filter_k), require_grad=True,
                                 is_trainable=True, is_parameter=True)

        if has_bias:
            bias_k = filter_k
            bias_shape = (out_channels, )
            self.bias = EagerNode(xuniform(bias_shape, bias_k), require_grad=True,
                                 is_trainable=True, is_parameter=True)
        else:
            self.bias = None

    def forward(self, x):
        return self.conv(x, self.filters, self.bias)

class MaxPool1D(ConvFunction):

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        if stride is None:
            self.stride = kernel_size
        else:
            self.stride = stride
        self.padding = padding

    def forward(self, x):
        '''
        :param x: (N, C, L)
        :return: (N, C, output_L)
        '''
        x = x.data
        N, C, L = x.shape

        # padding input
        if self.padding != 0:
            after_padding = np.zeros(shape=(N, C, L + 2 * self.padding),
                                     dtype=x.dtype)
            after_padding[:, :, self.padding: self.padding + L] = x
        else:
            after_padding = x

        # init output_template
        output_L = self.get_outputL(L, self.kernel_size,
                                    self.padding, self.stride)
        output_template = np.zeros(shape=(N, C, output_L))

        idx_list = []
        # calculate
        for L_i in range(output_L):
            ptr = L_i * self.stride

            # (N, C)
            mid_result = np.max(after_padding[:, :, ptr: ptr + self.kernel_size],
                                axis=2)
            idx_list.append( np.where(mid_result.reshape((N,C,1)) == after_padding[:, :, ptr: ptr + self.kernel_size]))
            output_template[:, :, L_i] = mid_result


        output = EagerNode(output_template, require_grad=True, is_trainable=True)
        self.process_output_node(output)
        self.save_for_backward['idx_list'] = idx_list
        self.save_for_backward['input_shape'] = x.shape
        self.save_for_backward['after_padding_shape'] = after_padding.shape
        return output

    def backward(self, grad_pre):
        #grad_pre (N, C, output_L)
        idx_list = self.save_for_backward['idx_list']
        _, _, L = self.save_for_backward['input_shape']
        _, _, output_L = grad_pre.shape
        grad_template = np.zeros(self.save_for_backward['after_padding_shape'], dtype=grad_pre.dtype)
        for L_i in range(output_L):
            ptr = L_i * self.stride
            grad_template[:, :, ptr : ptr + self.kernel_size][idx_list[L_i]] = grad_pre[:, :, L_i].reshape(-1)
        grad_template = grad_template[:, :, self.padding : self.padding + L]
        return grad_template

class MaxPool2D(ConvFunction):

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        ###
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple) or isinstance(kernel_size, list):
            self.kernel_size = kernel_size
        else:
            raise Exception('kernel size should be single int or tuple or list')

        ###
        if stride is None:
            self.stride = self.kernel_size
        else:
            if isinstance(stride, int):
                self.stride = (stride, stride)
            elif isinstance(stride, tuple) or isinstance(stride, list):
                self.stride = stride
            else:
                raise Exception('stride should be single int or tuple or list')

        ###
        if isinstance(padding, int):
            self.padding = (padding, padding)
        elif isinstance(padding, tuple) or isinstance(padding, list):
            self.padding = padding
        else:
            raise Exception('padding should be single int or tuple or list')

    def forward(self, x):
        '''
        :param x: (N, C, H, W)
        :return: (N, C, output_H, output_W)
        '''
        x = x.data
        N, C, H, W = x.shape

        # padding input
        if self.padding != 0:
            after_padding = np.zeros(shape=(N, C, H + 2 * self.padding[0], W + 2 * self.padding[1]),
                                     dtype=x.dtype)
            after_padding[:, :, self.padding[0]: self.padding[0] + H,
            self.padding[1]: self.padding[1] + W] = x
        else:
            after_padding = x

        # init output_template
        output_H = self.get_outputL(H, self.kernel_size[0],
                                    self.padding[0], self.stride[0])
        output_W = self.get_outputL(W, self.kernel_size[1],
                                    self.padding[1], self.stride[1])

        output_template = np.zeros(shape=(N, C, output_H, output_W))

        # calculate
        idx_list = []
        for H_i in range(output_H):
            H_ptr = H_i * self.stride[0]
            for W_i in range(output_W):
                W_ptr = W_i * self.stride[1]

                # (N, in_channels)
                mid_result = np.max(np.max(after_padding[:, :, H_ptr: H_ptr + self.kernel_size[0],
                             W_ptr: W_ptr + self.kernel_size[1]], axis=-1), axis=-1)

                idx_list.append( np.where(after_padding[:, :, H_ptr: H_ptr + self.kernel_size[0],
                             W_ptr: W_ptr + self.kernel_size[1]] == mid_result.reshape(N, C, 1, 1)) )

                output_template[:, :, H_i, W_i] = mid_result

        output = EagerNode(output_template, require_grad=True, is_trainable=True)
        self.process_output_node(output)
        self.save_for_backward['idx_list'] = idx_list
        self.save_for_backward['input_shape'] = x.shape
        self.save_for_backward['after_padding_shape'] = after_padding.shape
        return output

    def backward(self, grad_pre):
        # grad_pre (N, C, output_H, output_W)
        idx_list = self.save_for_backward['idx_list']
        N, C, H, W = self.save_for_backward['input_shape']
        _, _, output_H, output_W = grad_pre.shape
        grad_template = np.zeros(self.save_for_backward['after_padding_shape'], dtype=grad_pre.dtype)
        for H_i in range(output_H):
            H_ptr = H_i * self.stride[0]
            for W_i in range(output_W):
                W_ptr = W_i * self.stride[1]
                grad_template[:, :, H_ptr : H_ptr + self.kernel_size[0],
                W_ptr : W_ptr + self.kernel_size[1]][idx_list[H_i * W_i]] = grad_pre[:, :, H_i, W_i].reshape(-1)
                # grad_template[:, :, H_ptr: H_ptr + self.kernel_size[0],
                # W_ptr: W_ptr + self.kernel_size[1]][idx_list[H_i * W_i]] = 1

        grad_template = grad_template[:, :, self.padding[0] : self.padding[0] + H,
                        self.padding[1] : self.padding[1] + W]
        return grad_template

class BatchNorm(BaseFunction):

    def __init__(self, num_features, eps=1e-05, momentum=0.1
                 , affine=True):
        super().__init__()
        self.C = num_features

        self.momentum = EagerNode(np.array([momentum]), require_grad=False,
                             is_trainable=False, is_parameter=False)
        self.affine =affine
        if self.affine:
            self.gamma = EagerNode(np.ones((num_features,), dtype=np.float),
                                   require_grad=True, is_trainable=True,
                                   is_parameter=True)
            self.beta = EagerNode(np.zeros((num_features,), dtype=np.float),
                                   require_grad=True, is_trainable=True,
                                   is_parameter=True)

        self.running_mean = EagerNode(np.ones((num_features, 1, 1), dtype=np.float), require_grad=False,
                             is_trainable=False, is_parameter=False)
        self.running_v = EagerNode(np.zeros((num_features, 1, 1), dtype=np.float), require_grad=False,
                             is_trainable=False, is_parameter=False)

        self.eps = EagerNode(np.array([eps]), require_grad=False,
                             is_trainable=False, is_parameter=False)
        self.one_divide_two = EagerNode(np.array([1 / 2.]), require_grad=False,
                             is_trainable=False, is_parameter=False)
        self.momentum_0 = EagerNode(np.array([1. - momentum]), require_grad=False,
                             is_trainable=False, is_parameter=False)
    def forward(self, x):
        '''
        :param x: (N, C, L, ...)
        :return:
        '''
        x = x.data

        loop_time = len(x.shape) - 2

        if self.is_training:
            #update running mean (C,)
            mean = x.reduce_mean(axis=0, keepdims=False)
            for _ in range(loop_time):
                mean = mean.reduce_mean(axis=-1, keepdims=False)
            shape = (1, self.C) + (1,) * loop_time
            mean = mean.reshape(shape)
            self.running_mean = self.momentum_0 * self.running_mean + self.momentum * mean

            #update running v
            v = ((x - mean) ** 2).reduce_mean(axis=0)
            for _ in range(loop_time):
                v = v.reduce_mean(axis=-1, keepdims=False)
            shape = (1, self.C) + (1,) * loop_time
            v = v.reshape(shape)

            self.running_v = self.momentum_0 * self.running_v + self.momentum * v

        if self.affine:
            result = self.gamma * (x - self.running_mean) / \
                     (self.running_v + self.eps) ** (self.one_divide_two) + self.beta
        else:
            result = (x - self.running_mean) / (self.running_v + self.eps) ** (self.one_divide_two)

        output = EagerNode(result, require_grad=True, is_trainable=True)

        return output

