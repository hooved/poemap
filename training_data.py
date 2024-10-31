import os, math, copy, random, glob
from pathlib import Path
from collections import defaultdict
#from stream.client import draw_minimap, get_moves
import numpy as np
from PIL import Image
from helpers import shrink_with_origin, pad_to_square_multiple
from models import AttentionUNet
from typing import DefaultDict, List
from scipy.ndimage import convolve

class ViTDataLoader:
  def __init__(self, data_dir, test_samples_per_class=1, batch_size=64):
    self.data_dir, self.batch_size = data_dir, batch_size

    self.class_to_paths = defaultdict(set)
    patches = set(glob.glob(os.path.join(data_dir, "*", "*", "*.npz")))
    origins = set(glob.glob(os.path.join(data_dir, "*", "*", "*_origin.npz")))
    patches = patches - origins
    for fp in patches:
      self.class_to_paths[self.fp_to_class(fp)].add(fp)

    self.train_test_split(test_samples_per_class)
    self.shuffle_training_data()

  def fp_to_class(self, fp):
    return int(Path(fp).relative_to(Path(self.data_dir)).parts[0])

  def train_test_split(self, test_samples_per_class):
    # todo: for validation, don't use minimaps that only capture layout origin, which are unlikely to be predictive
    self.train_data = copy.deepcopy(self.class_to_paths)
    self.test_data = defaultdict(set)
    for layout in self.class_to_paths:
      for _ in range(test_samples_per_class):
        if self.train_data[layout]:
          self.test_data[layout].add(self.train_data[layout].pop())

  def shuffle_training_data(self):
    # we are being lazy and tracking filepaths, will load data into memory later
    # the unit of data for training/inference is a set of patches of a minimap
    # here, the patches are represented as a filepath string, later will load as np.ndarray
    self.training_batches: List[List[str]] = list()
    shuffled = {k: random.sample(list(v), len(v)) for k,v in self.train_data.items()}
    num_data = sum(len(val) for val in self.train_data.values())
    num_batches = math.ceil(num_data / self.batch_size)
    quotas = {k: math.ceil(len(v) / num_batches) for k,v in shuffled.items()}
    for _ in range(num_batches):
      self.training_batches.append(list())
      for layout in shuffled:
        # for class balance in training batches
        for __ in range(quotas[layout]):
          if shuffled[layout]:
            patches_fp = shuffled[layout].pop()
            self.training_batches[-1].append(patches_fp)

  def get_training_data(self, max_patches):
    for batch in self.training_batches:
      X = [np.load(fp)['data'][0] for fp in batch]
      # randomize patch order, it shouldn't matter because we calc position embeddings based on attached coords
      for i, patches in enumerate(X):
        if patches.shape[0] > max_patches:
          X[i] = patches[np.random.choice(patches.shape[0], size=max_patches, replace=False)]
        else:
          np.random.shuffle(patches)
      Y = [self.fp_to_class(fp) for fp in batch]
      yield X, Y

def find_unprocessed_frames(data_dir: str) -> DefaultDict[str, set]:
  not_done = defaultdict(set)
  for root, dirs, files in os.walk(data_dir):
    for file in files:
      prefix, ext = os.path.splitext(file)
      if ext == ".png" and prefix[-1] != 'o' and f"{prefix}.npz" not in files:
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
      minimap, origin = shrink_with_origin(minimap, origin)
      minimap, origin = extract_map_features(minimap, origin, model)
      o_id = get_frame_id(target_frame)
      Image.fromarray(minimap * 255, mode="L").save(os.path.join(instance, f"{o_id}o.png"))
      minimap = get_patches(minimap, origin)
      minimap = get_tokens(minimap)
      np.savez_compressed(os.path.join(instance, f"{o_id}.npz"), data=minimap)

def crop_to_content(image):
  white_pixels = np.argwhere(image == 1)
  #assert len(white_pixels) > 0
  if len(white_pixels) == 0:
    # TODO: handle empty images downstream
    return image, (0, 0)
  y_min, x_min = white_pixels.min(axis=0)
  y_max, x_max = white_pixels.max(axis=0)
  cropped_image = image[y_min:y_max+1, x_min:x_max+1]
  return cropped_image, (y_min, x_min)

def clean_sparse_pixels(image, threshold=3, neighborhood_size=8):
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
  mask[*origin] = 1
  minimap = np.concatenate([minimap, mask], axis=-1)
  minimap = pad_to_square_multiple(minimap, pad_multiple)
  origin = np.where(minimap[..., -1] == 1)
  origin = tuple(int(x[0]) for x in origin)
  minimap = minimap[:,:,0:-1].astype(np.uint8)
  return minimap, origin

def extract_map_features(minimap, origin, model, threshold, neighborhood_size):
  minimap, origin = pad_with_origin(minimap, origin, 32)
  pred = model.batch_inference(minimap, chunk_size=32)
  pred = clean_sparse_pixels(pred, threshold=threshold, neighborhood_size=neighborhood_size)
  #pred, offsets = crop_to_content(pred)
  #origin = tuple(int(val - offset) for val, offset in zip(origin, offsets))
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

# sort tokens in ascending order by euclidean distance from origin
def distance_sort(tokens):
  x = tokens[..., -2]
  y = tokens[..., -1]
  x2_y2 = x**2 + y**2
  x2_y2 = x2_y2[..., np.newaxis]
  tokens = np.concatenate((tokens, x2_y2), axis=-1)
  tokens = tokens[np.argsort(x2_y2[:,0,0,0])]
  # tokens now has new element at end of last axis: euclidean distance from origin
  return tokens

def tokenize_minimap(minimap: np.ndarray, origin: np.ndarray, model):
  # minimap: color image of shape (vertical, horizontal, 3)
  assert len(minimap.shape) == 3 and minimap.shape[2] == 3 and minimap.dtype == np.uint8
  # origin: y,x coordinates of layout entrance, y is pixels down from image top, x is pixels right from image left
  assert origin.shape == (2,) and origin.dtype == np.uint32
  minimap, origin = extract_map_features(minimap, origin, model, threshold=10, neighborhood_size=15)
  patches = get_patches(minimap, origin)
  tokens = get_tokens(patches)
  tokens = distance_sort(tokens)
  return tokens

if __name__=="__main__":

  model = AttentionUNet("AttentionUNet_4")
  model.load()
  prepare_training_data("data/train", model)
