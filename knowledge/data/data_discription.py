__author__ = 'sunmingming01'


import numpy as np

def make_dtype_from_str(discription_str):

    name_type_pairs = [elem.split(":") for elem in filter(None, discription_str.split())]
    name_type_pairs = [(name, np.dtype(type_str)) for name, type_str in name_type_pairs]
    return np.dtype(name_type_pairs)