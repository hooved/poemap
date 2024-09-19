import numpy as np
import cv2

def pad_to_square_multiple(array: np.ndarray, square_size: int):
    h, w, c = array.shape
    new_h = int(np.ceil(h / square_size) * square_size)
    new_w = int(np.ceil(w / square_size) * square_size)
    pad_h = new_h - h
    pad_w = new_w - w
    return np.pad(array, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

def shrink_image(img: np.ndarray, target_size: int):
    h, w = img.shape[:2]
    ratio = target_size / max(h, w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    #resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    return resized