import os, importlib, sys
from collections import defaultdict
from stream.client import draw_minimap, frames_to_map, get_moves
import numpy as np
from PIL import Image
from helpers import shrink_image, pad_to_square_multiple
from models import AttentionUNet
from typing import DefaultDict
from scipy.ndimage import convolve

def find_unprocessed_frames(data_dir: str) -> DefaultDict[str, set]:
  not_done = defaultdict(set)
  for root, dirs, files in os.walk(data_dir):
    for file in files:
      prefix, ext = os.path.splitext(file)
      if ext == ".png" and f"{prefix}.npz" not in files:
        #not_done[root].add(os.path.join(root, file))
        not_done[root].add(file)
  return not_done
  
get_frame_id = lambda x: int(os.path.splitext(x)[0])

# extract middle of 4k frame
# player icon is at 1920, 1060, in 4k
def crop_frame(frame: np.ndarray, box_radius: int):
  return frame[1060-box_radius: 1060+box_radius, 1920-box_radius: 1920+box_radius, :]

def prepare_training_data(data_dir, model):
  todo = find_unprocessed_frames(data_dir)
  for instance in todo:
    # don't process unneeded frames past max_frame_id
    max_frame_id = max(get_frame_id(f) for f in todo[instance])
    frames = [(get_frame_id(f), f) for f in os.listdir(instance)]
    # both frames and moves will be in chronological order due to sorting
    frames = sorted([(fid, f) for fid, f in frames if fid <= max_frame_id])
    frames = [(fid, crop_frame(np.array(Image.open(os.path.join(instance, f))), 600)) for fid, f in frames]
    # calculating moves is expensive, so we do it upfront once for all frames, then slice thereafter
    moves = get_moves([f for fid, f in frames])
    for target_frame in todo[instance]:
      tf_id = get_frame_id(target_frame)
      tf_idx = next((i for i, tpl in enumerate(frames) if tpl[0]==tf_id), -1)
      frames_subset = np.array([f for fid, f in frames][0:tf_idx + 1])
      minimap, origin = draw_minimap(frames_subset, moves[0:tf_idx + 1])

      # shrink image, recalculate origin
      dims = minimap.shape[0:2]
      max_dim_idx = dims.index(max(dims))
      new_size = dims[max_dim_idx] // 2
      origin = tuple(int(x * new_size / max(dims)) for x in origin)
      minimap = shrink_image(minimap, new_size)
      # pad image, use mask to track origin position
      mask = np.zeros((*minimap.shape[0:2], 1))
      mask[origin] = 1
      minimap = np.concatenate([minimap, mask], axis=-1)
      minimap = pad_to_square_multiple(minimap, 32)
      origin = np.where(minimap[..., 3] == 1)
      origin = tuple(int(x[0]) for x in origin)
      minimap = minimap[:,:,0:3].astype(np.uint8)
      clean, clean_origin = extract_map_features(minimap, origin, model)
      yield clean, clean_origin
      #yield minimap, origin
      #break
    break

def crop_to_content(image):
  white_pixels = np.argwhere(image == 1)
  assert len(white_pixels) > 0
  y_min, x_min = white_pixels.min(axis=0)
  y_max, x_max = white_pixels.max(axis=0)
  cropped_image = image[y_min:y_max+1, x_min:x_max+1]
  return cropped_image, (y_min, x_min)

def clean_sparse_pixels(image, threshold=3, neighborhood_size=3):
  # Create a kernel for counting neighbors
  kernel = np.ones((neighborhood_size, neighborhood_size))
  kernel[neighborhood_size//2, neighborhood_size//2] = 0  # Don't count the pixel itself
  # Count white neighbors for each pixel
  neighbor_count = convolve(image.astype(int), kernel, mode='constant')
  # Create a mask of pixels to keep (either black or with enough white neighbors)
  mask = (image == 0) | (neighbor_count >= threshold)
  # Apply the mask to the original image
  cleaned_image = image * mask
  return cleaned_image

def extract_map_features(image, origin, model):
  pred = model.batch_inference(image, chunk_size=32)
  pred = clean_sparse_pixels(pred, threshold=20, neighborhood_size=40)
  pred, offsets = crop_to_content(pred)
  origin = tuple(int(val - offset) for val, offset in zip(origin, offsets))
  return pred, origin

if __name__=="__main__":

  model = AttentionUNet("AttentionUNet_4")
  model.load()
  gen = prepare_training_data("data/train", model)
  minimap, origin = next(gen)

  done = 1

