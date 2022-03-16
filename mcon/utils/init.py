import numpy as np

def zero_init(shape, dtype=np.float):
    return np.zeros(shape, dtype)

def one_init(shape, dtype=np.float):
    return np.ones(shape, dtype)

def random_init(shape, *args, **kwargs):
    return np.random.random(shape)

def xuniform(shape, k):
    sqrt_k = np.sqrt(k)
    return np.random.uniform(-sqrt_k, sqrt_k, size=shape)