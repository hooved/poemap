import os, math, copy, random, glob
from pathlib import Path
from collections import defaultdict
#from stream.client import draw_minimap, get_moves
import numpy as np
from PIL import Image
from helpers import shrink_with_origin, pad_to_square_multiple
from models import AttentionUNet
from typing import DefaultDict, List, Dict, Tuple
from scipy.ndimage import convolve
import time
from tqdm import tqdm

"""
New dataloader:

# Prerequisite: convert layout.png to layout mask (mask.npz) by running this script, `python training_data.py`

- load filepaths to layout mask (npz) / paths (npz) pairs
- schedule epochs / steps, with class balance on each step
- each step specifies mask/path pairs which are subsets of the total
for each epoch:
  for each batch (step):
    for each mask/path pair in batch:
      load mask.npz -> np.ndarray
      load path.npz -> np.ndarray
      mask + path + mask origin -> patches, np.ndarray
  zero_grad, ViT(patches), backward, step

use multiprocessing to run training while preloading batches in parallel

- Load all mask filepaths, and paths into memory (small footprint)
--- load all mask origins: same as original image origins
- For each epoch, schedule mask fp/path pairs into steps, randomly but with class balance
- Only training process exists initially, and computes the schedule
- Complete training schedule is computed before training and sent to separate process (loader process)
- Loader process loads up to N steps of data in advance, which is ready immediately when needed by training process

############

Data sketch:

mask np.ndarray; shape (y, x)
mask_origin np.ndarray; shape (2,)
paths Dict[path_id str, path np.ndarray] 
path np.ndarray; shape (N_points, 2)

# batch of mask/path pairs
[
  (mask1, path1),
  (mask2, path2),
]

# each mask/path pair gives a set of N 32x32 pixel patches, last axis has 0/1 at 0th element, y,x coords at 1st/2nd elements
[(N_1,32,32,3), (N_2,32,32,3), ...]

# each patch set of N patches is padded with mask token to 128 patches (or truncated if N > 128)
Tensor(batch_size, 128,...)
"""

# mask-path pairs are the highest level unit of data for training
# each mask-path pair is converted to a token sequence for ViT forward pass
class MaskPath:
  def __init__(self, layout_id: int, mask: np.ndarray, origin: np.ndarray, path: np.ndarray):
    self.layout_id, self.mask, self.origin, self.path = layout_id, mask, origin, path
    self.tokens = None

  def load(self):
    if self.tokens is not None: return self.tokens
    revealed = explore_map(self.mask, self.path)
    #Image.fromarray(revealed * 255, mode="L").save("mp1.png")
    self.tokens = tokenize_mask(revealed, self.origin)
    return self.tokens

class ViTDataLoader:
  def __init__(self, data_dir):
    self.data_dir = data_dir
    self.data = self.load_maskpath_db()
    self.train_test_split()

  def load_maskpath_db(self, paths_handle:str="paths.npz"):
    # We are using precomputed masks
    masks = set(glob.glob(os.path.join(self.data_dir, "*", "*", "*_mask.png"))) 
    data = defaultdict(lambda: defaultdict(list))

    for mask_fp in tqdm(masks, desc="Processing masks", unit="mask"):
      # organize data into hierarchy for balanced layout class sampling
      # layout_dir > instance_dir > mask/origin/path
      instance_dir = os.path.dirname(mask_fp)

      paths_fp = os.path.join(instance_dir, paths_handle)
      if os.path.exists(paths_fp):
        paths = list(np.load(paths_fp, allow_pickle=True)['paths'])
        assert len(paths) > 0
      else: continue

      layout_dir = os.path.dirname(instance_dir)
      instance_id = int(os.path.relpath(instance_dir, layout_dir))
      layout_id = int(os.path.relpath(layout_dir, self.data_dir))

      # currently masks were saved with values of 0 or 255 for easy visualization
      mask = (np.array(Image.open(mask_fp)) / 255).astype(np.uint8)
      assert len(mask.shape) == 2
      # TODO: clean up filename patterns, should just be loading instance_dir/origin.npz
      #origin = np.load(os.path.join(instance_dir, ""))
      origin = np.load(f"{mask_fp.split('_mask.png')[0]}_origin.npz")['data']
      assert len(origin.shape) == 1 and origin.shape[0] == 2
      origin = tuple(int(x) for x in origin)

      for path in tqdm(paths, desc=f"Paths for mask {os.path.basename(mask_fp)}", leave=False, unit="path"):
        assert len(path.shape) == 2
        assert path.shape[1] == 2
        mp = MaskPath(layout_id, mask, origin, path)
        # TODO: load later with multiprocessing if time consuming
        mp.load()
        data[layout_id][instance_id].append(mp)
    return data

  # split along layout instance mask axis (which is a higher level than paths through a layout instance)
  def train_test_split(self):
    """
    self.train_data = self.data.copy()
    self.test_data = defaultdict(lambda: defaultdict(list))
    for layout_id in self.train_data:
      for _ in range(self.test_layout_split):
        instance_id = random.choice(list(self.train_data[layout_id].keys()))
        instance_data = self.train_data[layout_id].pop(instance_id)
        self.test_data[layout_id][instance_id] = instance_data
    """
    # hand curate test data for now
    # an expert human would correctly classify these test data
    # an expert human wouldn't necessarily be able to classify all training data, which is more random
    self.train_data = self.data.copy()
    self.test_data = self.load_maskpath_db(paths_handle="paths_test.npz")

  def get_epoch(self, min_samples_per_class_per_step=1) -> Tuple[List[List]]:
    layout_to_samples = flatten_instances(self.train_data)
    for samples in layout_to_samples.values(): random.shuffle(samples)
    
    # assume some level of class balance in training data
    samples_per_layout = count_samples(layout_to_samples)
    assert max(samples_per_layout) / min(samples_per_layout) < 2

    num_steps = min(samples_per_layout) // min_samples_per_class_per_step
    assert num_steps > 0, f"not enough samples ({min(samples_per_layout)}) to meet quota for min_samples_per_class_per_step"
    # divide epoch into steps
    X = [[] for _ in range(num_steps)]

    while sum(count_samples(layout_to_samples)) > 0:
      for step in X:
        for layout_samples in layout_to_samples.values():
          if layout_samples:
            step.append(layout_samples.pop())

    Y = []
    for step in X:
      Y.append([sample.layout_id for sample in step])
    for step in X:
      for i, sample in enumerate(step):
        assert sample.tokens is not None, "realize tokens with MaskPath.load()"
        step[i] = sample.tokens
    return X, Y

  def get_test_data(self):
    layout_to_samples = flatten_instances(self.test_data)
    X = []
    for samples in layout_to_samples.values():
      X += samples
    Y = [sample.layout_id for sample in X]
    X = [sample.tokens for sample in X]
    return X, Y

# layout > instance > mask/path ---> layout > mask/path
def flatten_instances(l_i_mp):
  l_mp = defaultdict(list)
  for layout in l_i_mp:
    for instance in l_i_mp[layout]:
      l_mp[layout] += l_i_mp[layout][instance]
  return l_mp

def count_samples(layout_to_samples: Dict[int, List]):
  return list(len(v) for v in layout_to_samples.values())
    
get_frame_id = lambda x: int(os.path.splitext(x)[0])

# extract middle of 4k frame
# player icon is at 1920, 1060, in 4k
def crop_frame(frame: np.ndarray, box_radius: int):
  return frame[1060-box_radius: 1060+box_radius, 1920-box_radius: 1920+box_radius, :]

def prepare_training_data(data_dir, model):
  # compute tokens for complete layouts
  pngs = set(glob.glob(os.path.join(data_dir, "*", "*", "*.png")))
  masks = set(glob.glob(os.path.join(data_dir, "*", "*", "*_mask.png")))
  layouts = sorted(list(pngs - masks))
  for layout in layouts:
    minimap = np.array(Image.open(layout))
    wd = os.path.dirname(layout)
    num = os.path.splitext(os.path.basename(layout))[0]
    origin = np.load(os.path.join(wd, f"{num}_origin.npz"))['data']
    tokens, mask = tokenize_minimap(minimap, origin, model)
    # save mask of map features for visualization
    Image.fromarray(mask * 255, mode="L").save(os.path.join(wd, f"{num}_mask.png"))
    np.savez_compressed(os.path.join(wd, f"{num}.npz"), data=tokens)

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
  padded_origin = np.where(minimap[..., -1] == 1)
  padded_origin = tuple(int(x[0]) for x in padded_origin)
  minimap = minimap[:,:,0:-1].astype(np.uint8)
  # origin shouldn't change in current implementation
  assert tuple(int(x) for x in origin) == padded_origin
  return minimap, padded_origin

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
  return tokens, minimap

# Simulate exploring fog of war on a map, following given path

# light_radius: real in game radius is an oval, height from center = 120 px, width from center = 140 px (4k resolution)
# Our masks are downsampled by 2x in each dimension from 4k, so that would be vertical_radius = 60, horizontal_radius = 70
# For simplicity, use circular light radius of 65 px when sampling mask with path
def explore_map(no_fog_map: np.ndarray, path: np.ndarray, light_radius=65):
  # add fog of war over entire map
  revealed = np.zeros_like(no_fog_map)
  lr_squared = light_radius**2
  # For each path point, reveal map features within light_radius
  for y, x in path:
    # Define the bounding box for the circle
    x_min = max(x - light_radius, 0)
    x_max = min(x + light_radius + 1, no_fog_map.shape[1])
    y_min = max(y - light_radius, 0)
    y_max = min(y + light_radius + 1, no_fog_map.shape[0])

    # Create grid of coordinates for the bounding box
    yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        
    # Compute mask of points within the circular radius
    mask = (yy - y)**2 + (xx - x)**2 <= lr_squared

    # Update the revealed map using the mask
    revealed[y_min:y_max, x_min:x_max] = np.maximum(
        revealed[y_min:y_max, x_min:x_max],  # Keep previously revealed values
        no_fog_map[y_min:y_max, x_min:x_max] * mask
    )
  return revealed

def tokenize_mask(mask: np.ndarray, origin: np.ndarray):
  patches = get_patches(mask, origin)
  tokens = get_tokens(patches)
  return tokens

if __name__=="__main__":

  #model = AttentionUNet("AttentionUNet8_8600", depth=3).load()
  #prepare_training_data("data/train", model)
  dl = ViTDataLoader("data/train")
  for _ in range(10):
    t1 = time.perf_counter()
    X, Y = dl.get_training_data()
    print(f"elapsed: {(time.perf_counter() - t1):0.3f}")

  done = 1
