import numpy as np

def pad_to_square_multiple(array, square_size):
    h, w, c = array.shape
    new_h = int(np.ceil(h / square_size) * square_size)
    new_w = int(np.ceil(w / square_size) * square_size)
    pad_h = new_h - h
    pad_w = new_w - w
    return np.pad(array, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')