import sys

sys.path.append('./mcon')

import numpy as np

# '''
# 1.test gradient
# '''
# from mcon.node.base_node import Node
# from mcon.function.functions import *
# a = Node(np.ones((1,2,3)), require_grad=True, is_trainable=True)
# b = Node(np.ones((2,1,1)), require_grad=True, is_trainable=True)
#
# result0 = Add()(a, b)
#
# c = Node(np.ones((2,1,1)), require_grad=True, is_trainable=True)
# result1 = Add()(c, result0)
#
# result = Add()(a, result1)
# result.backward(np.array([ [[1,0,0],[0,0,1]]
#                              ,[[1,2,3],[199,9,1]]]))
#
# result.backward(np.array([ [[1,0,0],[0,0,1]]
#                              ,[[1,2,3],[199,9,1]]]))

# '''
# 2.***test operator overload
# '''
# # from mcon.node.eager_node import EagerNode as Node
# # a = Node(np.ones((1,2,3)), require_grad=True, is_trainable=True)
# # b = Node(np.ones((2,1,1)), require_grad=True, is_trainable=True)
# #
# # result0 = a+b
# #
# # c = Node(np.ones((2,1,1)), require_grad=True, is_trainable=True)
# # result1 = c+result0
# #
# # result = a+result1
# # result.backward(np.array([ [[1,0,0],[0,0,1]]
# #                              ,[[1,2,3],[199,9,1]]]))
# #
# # result.backward(np.array([ [[1,0,0],[0,0,1]]
# #                              ,[[1,2,3],[199,9,1]]]))

# '''
# # 2.***test operator overload * /
# # '''
# from mcon.node.eager_node import EagerNode as Node
# a = Node(np.ones((1,2,3), dtype=np.float), require_grad=True, is_trainable=True)
# b = Node(np.ones((2,1,1), dtype=np.float)+1, require_grad=True, is_trainable=True)
#
# result0 = a * b
#
# # c = Node(np.ones((2,1,1)), require_grad=True, is_trainable=True)
# # result1 = c+result0
#
# result = a + result0
# result.backward(np.array([ [[1,0,0],[0,0,1]]
#                              ,[[1,2,3],[199,9,1]]], dtype=np.float))
# print()


# '''
# # 3.***test operator overload Reshape, pow
# # '''
# from mcon.node.eager_node import EagerNode as Node
# a0 = Node(np.ones((1,3,2), dtype=np.float), require_grad=True, is_trainable=True)
# b = Node(np.ones((2,1,1), dtype=np.float)+1, require_grad=True, is_trainable=True)
# a = a0.reshape((1,2,3))
# result0 = a * b
#
# c = Node(np.array([1.]), require_grad=True, is_trainable=True)
#
# result = a**c + result0
# result.backward(np.array([ [[1,0,0],[0,0,1]]
#                              ,[[1,2,3],[199,9,1]]], dtype=np.float))
# print()

# '''
# # 4.***test operator overload Dot and Add
# # '''
# from mcon.node.eager_node import EagerNode as Node
# a = Node(np.ones((1,3), dtype=np.float), require_grad=True, is_trainable=True)
# b = Node(np.ones((3,2), dtype=np.float)+1, require_grad=True, is_trainable=True)
# c = Node(np.ones((2,), dtype=np.float)+1, require_grad=True, is_trainable=True)
# d = Node(np.ones((2,1), dtype=np.float)+1, require_grad=True, is_trainable=True)
# e = Node(np.ones((1,), dtype=np.float)+1, require_grad=True, is_trainable=True)
# result = (a @ b + c) @ d + e
#
# result.backward(np.array([[9.]], dtype=np.float))
# print()


# '''
#  5.***test Linear
# '''
# from mcon.node.eager_node import EagerNode as Node
# from mcon.function.modules import Linear
#
# bsz = 1
# in_features = 50
# out_features = 1
# linear = Linear(in_features, out_features)
# a = Node(np.ones((bsz,in_features), dtype=np.float), require_grad=True, is_trainable=True)
#
# result = linear(a)
# result.backward(np.array([[9.]*bsz], dtype=np.float))
# print(linear.parameter())

# '''
#  5.***test First Model
# '''
# from mcon.node.eager_node import EagerNode as Node
# from mcon.function.modules import Linear
# from mcon.function.base_function import BaseFunction
#
# class FirstNet(BaseFunction):
#
#     def __init__(self, in_features, mid_features):
#         super().__init__()
#         self.linear1 = Linear(in_features, mid_features)
#         self.linear2 = Linear(mid_features, 1)
#
#     def forward(self, x):
#         y0 = self.linear1(x)
#         y = self.linear2(y0)
#         return y
#
# bsz = 100
# in_features = 1000
# mid_features = 500
# model = FirstNet(in_features, mid_features)
# x = Node(np.random.random((bsz, in_features)), require_grad=True)
# result = model(x)
# result.backward(np.ones((bsz,1), dtype=np.float))
# print(model.parameter())
#
# from mcon.utils.utils import save_model, load_model
# save_model(model, 'first_model.pkl')
# model = load_model('first_model.pkl')
# print(model.parameter())



# '''
#  5.***test Loss Function, Activation
# '''
# from mcon.node.eager_node import EagerNode as Node
# from mcon.function.modules import Linear
# from mcon.function.modules import MSELoss, CrossEntropyLoss, ReLU, Sigmoid
# from mcon.function.base_function import BaseFunction
#
# class FirstNet(BaseFunction):
#
#     def __init__(self, in_features, mid_features):
#         super().__init__()
#         self.linear1 = Linear(in_features, mid_features)
#         self.relu = ReLU()
#         self.linear2 = Linear(mid_features, 2)
#         self.sigmoid = Sigmoid()
#
#     def forward(self, x):
#         a = Node(np.random.random((1,)) * 10000, require_grad=False)
#         b = Node(np.random.random((1,)) * 10, require_grad=False)
#         y0 = self.relu(self.linear1(x) / a)
#         y = self.sigmoid(self.linear2(y0)/ b)
#         return y
#
# bsz = 100
# in_features = 1000
# mid_features = 500
# model = FirstNet(in_features, mid_features)
# x = Node(np.random.random((bsz, in_features)), require_grad=True)
# output = model(x)
# target = Node(np.random.random((bsz,2)), require_grad=True)
#
# mse_loss = MSELoss()
# loss = mse_loss(output, target)
# loss.backward(1.)
# print()
#
# target = Node(np.random.randint(0 ,2, size=(bsz,1), dtype=np.longlong), require_grad=True)
# ce_loss = CrossEntropyLoss()
# loss = ce_loss(output, target)
# loss.backward(1.)
# print()

# '''
#  6.***test optimizer
# '''
# from mcon.node.eager_node import EagerNode as Node
# from mcon.function.modules import Linear
# from mcon.function.modules import MSELoss, CrossEntropyLoss, ReLU, Sigmoid
# from mcon.function.base_function import BaseFunction
# from mcon.optim import Adam, SGD, AdaGrad
# from mcon.function.modules import Softmax
#
# class FirstNet(BaseFunction):
#
#     def __init__(self, in_features, mid_features):
#         super().__init__()
#         self.linear1 = Linear(in_features, mid_features)
#         self.relu = ReLU()
#         self.linear2 = Linear(mid_features, 2)
#         self.sigmoid = Sigmoid()
#         self.softmax = Softmax()
#
#     def forward(self, x):
#         a = Node(np.random.random((1,)) * 10000, require_grad=False)
#         b = Node(np.random.random((1,)) * 10, require_grad=False)
#         y0 = self.relu(self.linear1(x) / a)
#         y = self.softmax(self.linear2(y0)/ b)
#         return y
#
# epochs = 100
# bsz = 100
# in_features = 20
# mid_features = 50
#
# model = FirstNet(in_features, mid_features)
#
# x = Node(np.random.random((bsz, in_features)),
#          require_grad=True)
# target = Node(np.random.random((bsz,2)), require_grad=True)
#
# mse_loss = MSELoss()
#
# optimizer = Adam(model.parameter())
# print(model.parameter())
# for epoch in range(epochs):
#     output = model(x)
#     loss = mse_loss(output, target)
#
#     optimizer.zero_grad()
#     loss.backward(1.)
#     optimizer.step()
#     print('Epoch:{} loss:{}'.format(epoch, loss.data))
# print()
#
# model = FirstNet(in_features, mid_features)
# x = Node(np.random.random((bsz, in_features)),
#          require_grad=True)
# target = Node(np.random.randint(0 ,2, size=(bsz,1), dtype=np.longlong), require_grad=True)
# ce_loss = CrossEntropyLoss()
# optimizer = Adam(model.parameter())
# for epoch in range(epochs):
#     output = model(x)
#     loss = ce_loss(output, target)
#
#     optimizer.zero_grad()
#     loss.backward(1.)
#     optimizer.step()
#     print('Epoch:{} loss:{}'.format(epoch, loss.data))
# print()
#


# '''
# # 7.***test operator overload Exp reduce_mean reduce_sum
# # '''
# from mcon.node.eager_node import EagerNode as Node
# a = Node(np.ones((1,5), dtype=np.float), require_grad=True, is_trainable=True)
#
# result = a.reduce_sum().exp()
#
# result.backward(np.array([9.], dtype=np.float))
# print(result.data)



# '''
# # 7.***test Softmax
# Here is an interesting phenomenon, if you call result.backward(np.array([[9,9,9,9,9]], dtype=np.float)),
# the gradient of a is [[0.,0.,0.,0.,0.]]. This phenomenon can be expained. If you are interested about it,
# you can debug the backward propagation, and you will find that exp_x.grad is already all 0.
# # '''
# from mcon.node.eager_node import EagerNode as Node
# from mcon.function.modules import Softmax
# a = Node(np.random.random((1,5)), require_grad=True, is_trainable=True)
#
# result = Softmax()(a)
#
# result.backward(np.array([[9.,0,3,0,0]], dtype=np.float))
# print(result.data)

'''
# 8.***test Conv1D
# '''
from mcon.node.eager_node import EagerNode as Node
from mcon.function.modules import Softmax, Conv1D
a = Node(np.random.random((2,5,3)), require_grad=True, is_trainable=True)
a = a ** 0
result = Conv1D(in_channels=5, out_channels=6, kernel_size=3, stride=1,
                 padding=1)(a)

result.backward(np.ones_like(result.data))
print(result.data)



# '''
# # 9.***test Conv2D
# # '''
# from mcon.node.eager_node import EagerNode as Node
# from mcon.function.modules import Conv2D
# a = Node(np.random.random((2,5,3,4)), require_grad=True, is_trainable=True)
#
# result = Conv2D(in_channels=5, out_channels=6, kernel_size=3, stride=1,
#                  padding=1)(a)
#
# result.backward(np.ones_like(result.data))
# print(result.data)

# '''
# # 10.***test Pool
# # '''
# from mcon.node.eager_node import EagerNode as Node
# from mcon.function.modules import MaxPool2D
# a = Node(np.random.random((2,5,3,4)), require_grad=True, is_trainable=True)
#
# result = MaxPool2D(kernel_size=2,
#                  padding=1)(a)
#
# result.backward(np.ones_like(result.data)*9)
# print(result.data)

# from mcon.node.eager_node import EagerNode as Node
# from mcon.function.modules import MaxPool1D
# a = Node(np.random.random((2,5,3)), require_grad=True, is_trainable=True)
#
# result = MaxPool1D(kernel_size=2,
#                  padding=1)(a)
#
# result.backward(np.ones_like(result.data)*9)
# print(result.data)

# '''
# # 11.***test mnist
# # '''
# from mcon.node.eager_node import EagerNode as Node
# from mcon.function.modules import Linear
# from mcon.function.modules import MSELoss, CrossEntropyLoss, ReLU, Sigmoid
# from mcon.function.base_function import BaseFunction
# from mcon.optim import Adam, SGD, AdaGrad
# from mcon.function.modules import Softmax
#
# class FirstNet(BaseFunction):
#
#     def __init__(self, in_features, mid_features):
#         super().__init__()
#         self.linear1 = Linear(in_features, mid_features)
#         self.relu = ReLU()
#         self.linear2 = Linear(mid_features, 10)
#         self.sigmoid = Sigmoid()
#         self.softmax = Softmax()
#
#     def forward(self, x):
#         y0 = self.relu(self.linear1(x))
#         y = self.sigmoid(self.linear2(y0))
#         return y
#
# epochs = 200
# bsz = 100
# in_features = 28*28
# mid_features = 500
# model = FirstNet(in_features, mid_features)
# #
# def load_mnist(mnist_image_file, mnist_label_file):
#     with open(mnist_image_file, 'rb') as f1:
#         image_file = np.frombuffer(f1.read(), np.uint8, offset=16).reshape(-1, 28*28)
#     with open(mnist_label_file, 'rb') as f2:
#         label_file = np.frombuffer(f2.read(), np.uint8, offset=16)
#
#     return image_file, label_file
#
# image_file, label_file = load_mnist('./mnist/train-images.idx3-ubyte',
#            './mnist/train-labels.idx1-ubyte')
#
# x = Node(image_file[:1000].astype(np.float) / 255., require_grad=False)
# target = Node(label_file.astype(np.longlong)[:1000].reshape(1000), require_grad=False)
#
# ce_loss = CrossEntropyLoss()
# optimizer = Adam(model.parameter(), lr=0.01)
# for epoch in range(epochs):
#     output = model(x)
#     loss = ce_loss(output, target)
#
#     optimizer.zero_grad()
#     loss.backward(1.)
#     optimizer.step()
#     print('Epoch:{} loss:{}'.format(epoch, loss.data))


