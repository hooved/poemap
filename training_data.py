import os, importlib, sys, math
from pathlib import Path
from collections import defaultdict
from stream.client import draw_minimap, frames_to_map, get_moves
import numpy as np
from PIL import Image
from helpers import shrink_image, pad_to_square_multiple
from models import AttentionUNet
from typing import DefaultDict
from scipy.ndimage import convolve

class ViTDataLoader:
  def __init__(self, data_dir):
    self.data_dir = data_dir
    self._class_to_paths = None

  @property
  def class_to_paths(self):
    if self._class_to_paths is None:
      self._class_to_paths = defaultdict(set)
      for root, dirs, files in os.walk(self.data_dir):
        for file in files:
          prefix, ext = os.path.splitext(file)
          if ext == ".npz":
            fp = os.path.join(root, file)
            class_id = int(Path(fp).relative_to(Path(self.data_dir)).parts[0])
            self._class_to_paths[class_id].add(fp)

    return self._class_to_paths

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
      minimap, origin = extract_map_features(minimap, origin, model)
      o_id = get_frame_id(target_frame)
      Image.fromarray(minimap * 255, mode="L").save(os.path.join(instance, f"{o_id}o.png"))
      minimap = get_patches(minimap, origin)
      minimap = get_tokens(minimap)
      np.savez_compressed(os.path.join(instance, f"{o_id}.npz"), data=minimap)

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

def pad_with_origin(minimap, origin, pad_multiple):
  # pad image, use mask to track origin position
  mask = np.zeros((*minimap.shape[0:2], 1))
  mask[origin] = 1
  minimap = np.concatenate([minimap, mask], axis=-1)
  minimap = pad_to_square_multiple(minimap, pad_multiple)
  origin = np.where(minimap[..., -1] == 1)
  origin = tuple(int(x[0]) for x in origin)
  minimap = minimap[:,:,0:-1].astype(np.uint8)
  return minimap, origin

def extract_map_features(image, origin, model):
  # shrink image, recalculate origin
  dims = image.shape[0:2]
  max_dim_idx = dims.index(max(dims))
  new_size = dims[max_dim_idx] // 2
  origin = tuple(int(x * new_size / max(dims)) for x in origin)
  minimap = shrink_image(image, new_size)
  minimap, origin = pad_with_origin(minimap, origin, 32)
  pred = model.batch_inference(minimap, chunk_size=32)
  pred = clean_sparse_pixels(pred, threshold=20, neighborhood_size=40)
  pred, offsets = crop_to_content(pred)
  origin = tuple(int(val - offset) for val, offset in zip(origin, offsets))
  return pred, origin

# Chunk the map into square patches, label each patch with y,x positions relative to origin
# We will use the y,x positions for token position embeddings
def get_patches(array, origin, ps=32):
  assert len(array.shape) == 2
  Y, X = array.shape
  # calc num patches in each direction from origin
  # make partial patches complete with empty padding
  y, x = origin
  up, down = math.ceil(y/ps), math.ceil((Y-y)/ps)
  pad_up, pad_down = (up*ps - y), (down*ps - (Y-y))
  y += pad_up
  left, right = math.ceil(x/ps), math.ceil((X-x)/ps)
  pad_left, pad_right = (left*ps - x), (right*ps - (X-x))
  x += pad_left
  patches = np.pad(array, ((pad_up, pad_down), (pad_left, pad_right)), mode='constant')

  # calc patch y,x dims for each pixel, relative to origin patch
  indices = np.indices(patches.shape).transpose(1,2,0)
  indices = indices // ps - np.array([up, left])
  patches = patches.reshape(*patches.shape, 1)
  patches = np.concatenate([patches, indices], axis=-1)
  return patches

# Remove completely black patches
def get_tokens(patches):
  Y,X = patches.shape[0:2]
  y_patches, x_patches = Y // 32, X // 32
  tokens = []
  for i in range(y_patches):
    for j in range(x_patches):
      patch = patches[i*32 : (i+1)*32, j*32 : (j+1)*32]
      if np.any(patch[:,:,0] > 0):
        tokens.append(patch)
  return np.array(tokens)

if __name__=="__main__":

  model = AttentionUNet("AttentionUNet_4")
  model.load()
  prepare_training_data("data/train", model)
