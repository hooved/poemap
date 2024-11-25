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
from tinygrad import Tensor
from multiprocessing import Process, Queue
from dataclasses import dataclass

"""
ideas to further improve vit (achieves ~75% accuracy with challenging test cases; random is 1/9 = 11%):
- identify failing test cases, add more paths to training dataset based on failing test cases
- dropout at classification stage

- jitter position coordinates or embeds (be consistent in x and y directions for all patches)
- dropout and jitter paths
- jitter individual pixels in masks
- deeper/more patch embedding params
- transformer block params

- acquire more raw layout instances
- improve minimap masking by UNet
"""

# mask-path pairs are the highest level unit of data for training
# each mask-path pair is converted to a token sequence for ViT forward pass
class MaskPath:
  def __init__(self, layout_id: int, instance_id: int, path_id: int, 
               mask: np.ndarray, origin: np.ndarray, path: np.ndarray, embed_dim: int):

    self.layout_id, self.instance_id, self.path_id = layout_id, instance_id, path_id
    self.mask, self.origin, self.path, self.embed_dim = mask, origin, path, embed_dim

    self.tokens, self.pe = None, None # lazy loading; these are realized later

  def load(self):
    if self.tokens is not None and self.pe is not None: return self.tokens, self.pe
    revealed = explore_map(self.mask, self.path)
    #Image.fromarray(revealed * 255, mode="L").save("mp1.png")
    self.tokens = tokenize_mask(revealed, self.origin)
    self.tokens, self.pe = tensorize_tokens(self.tokens, self.embed_dim)
    return self.tokens, self.pe

@dataclass(frozen=True)
class RealSample:
  layout_id: int
  tokens: Tensor
  pe: Tensor

class ViTDataLoader:
  def __init__(self, data_dir, test_dir, embed_dim=256):
    self.data_dir, self.test_dir, self.embed_dim = data_dir, test_dir, embed_dim
    self.all_data_realized = False
    self.data = self.load_maskpath_db()
    self.train_test_split()

  def load_maskpath_db(self, paths_handle:str="paths.npz"):
    # We are using precomputed masks
    masks = set(glob.glob(os.path.join(self.data_dir, "*", "*", "*_mask.png"))) 
    data = defaultdict(lambda: defaultdict(list))

    for mask_fp in tqdm(masks, desc=f"Lazy loading mask/path pairs for handle: {paths_handle}", unit="mask"):
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

      for path_id, path in enumerate(tqdm(paths, desc=f"Paths for mask {os.path.basename(mask_fp)}", leave=False, unit="path")):
        assert len(path.shape) == 2
        assert path.shape[1] == 2
        mp = MaskPath(layout_id, instance_id, path_id, mask, origin, path, self.embed_dim)
        # load later with a child process
        #mp.load()
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
    self.train_data = self.data.copy()
    #self.test_data = self.load_maskpath_db(paths_handle="paths_test.npz")

    # current training data is not identical to test time data; training data is synthetic due to drawing paths in MSPaint to simulate real paths
    # current test data is identical to what the user will be feeding to the model

    self.test_data = self.load_test_data()

  def schedule_epoch(self, min_samples_per_class_per_step=1) -> Tuple[List[List]]:
    layout_to_samples = flatten_instances(self.train_data)
    for samples in layout_to_samples.values(): random.shuffle(samples)
    
    # assume some level of class balance in training data
    samples_per_layout = count_samples(layout_to_samples)
    assert max(samples_per_layout) / min(samples_per_layout) < 2

    num_steps = min(samples_per_layout) // min_samples_per_class_per_step
    assert num_steps > 0, f"not enough samples ({min(samples_per_layout)}) to meet quota for min_samples_per_class_per_step"
    # divide epoch into steps
    X_steps = [[] for _ in range(num_steps)]

    while sum(count_samples(layout_to_samples)) > 0:
      for X in X_steps:
        for layout_samples in layout_to_samples.values():
          if layout_samples:
            X.append(layout_samples.pop())

    Y_steps = []
    for X in X_steps:
      Y_steps.append(Tensor([sample.layout_id for sample in X], requires_grad=False))
    return X_steps, Y_steps

  def _parallel_realize(self, X_steps: List[List[MaskPath]]):
    realized_q = Queue()
    realizer_p = Process(target=realizer_proc, args=(X_steps, realized_q,))
    realizer_p.start()
    try:
      for i, _ in enumerate(X_steps):
        i_X = realized_q.get()
        assert (X := i_X.get(i)), "mismatch when receiving data from parallel loader"
        if isinstance(X, dict) and (e := X.get("error")):
          raise Exception(e)
        else:
          # MaskPath realization occurred in child process; parent process objects are not realized
          # replace parent process's unrealized objects with realized ones, so future epochs don't need to reload data
          # TODO: this data is currently stored in GPU memory; if needed, move to host memory or unload/reload
          for mp in X:
            parent_mp = self.data[mp.layout_id][mp.instance_id][mp.path_id]
            parent_mp.tokens, parent_mp.pe = mp.tokens, mp.pe
          yield X # should now be equivalent to X_steps[i]
    finally:
      realizer_p.terminate()
      realizer_p.join()

  def get_epoch(self, min_samples_per_class_per_step=5):
    X_steps, Y_steps = self.schedule_epoch(min_samples_per_class_per_step)
    # pad for consistent batch size, for jit acceleration
    target_bs = max(len(step) for step in X_steps)

    if not self.all_data_realized:
      for i, X in enumerate(self._parallel_realize(X_steps)):
        Y = Y_steps[i]
        yield pad_batch(maskpaths_to_tensors(X), Y, target_bs)

      self.all_data_realized = True

    else:
      for X, Y in zip(X_steps, Y_steps):
        yield pad_batch(maskpaths_to_tensors(X), Y, target_bs)

  def load_test_data(self, token_cutoff=50) -> List[RealSample]:
    test_tokens = set(glob.glob(os.path.join(self.test_dir, "*", "*", "*_tokens.npz"))) 
    test_data = []
    for tokens_fp in test_tokens:
      layout_dir = os.path.dirname(os.path.dirname(tokens_fp))
      layout_id = int(os.path.relpath(layout_dir, self.test_dir))
      tokens, pe = tensorize_tokens(np.load(tokens_fp)['data'], self.embed_dim)
      test_data.append(RealSample(layout_id, tokens, pe))
    return test_data

  def get_test_data(self):
    assert self.test_data is not None
    X = [(sample.tokens, sample.pe) for sample in self.test_data]
    Y = Tensor([sample.layout_id for sample in self.test_data], requires_grad=False)
    return X, Y

  def ___old_get_test_data(self):
    layout_to_samples = flatten_instances(self.test_data)
    X: List[MaskPath] = []
    for samples in layout_to_samples.values():
      X += samples
    Y = Tensor([sample.layout_id for sample in X], requires_grad=False)
    # realize
    X = [sample.load() for sample in X]
    return X, Y

maskpaths_to_tensors = lambda X: [mp.load() for mp in X]

def pad_batch(X: List[Tuple[Tensor]], Y: Tensor, target_bs: int):
  num_pad_samples = target_bs - len(X)
  batch_pad_mask = Tensor(len(X) * [1.], requires_grad=False)
  if num_pad_samples:
    pad_tokens = Tensor.zeros(X[0][0].shape, requires_grad=False)
    pad_pe = Tensor.zeros(X[0][1].shape, requires_grad=False)
    pad_X = num_pad_samples * [(pad_tokens, pad_pe)]
    X = X + pad_X

    pad_Y = num_pad_samples * [0]
    Y = Y.cat(Tensor(pad_Y))
    Y.requires_grad = False

    batch_pad_mask = batch_pad_mask.cat(Tensor(num_pad_samples * [0.]))
    batch_pad_mask.requires_grad = False
  return X, Y, batch_pad_mask

def realizer_proc(X_steps: List[List[MaskPath]], realized_q: Queue):
  try:
    for i, X in enumerate(tqdm(X_steps, desc="Realizing steps", unit="step")):
      for sample in X: 
        sample.load()
      realized_q.put({i: X})

  except Exception as e:
    realized_q.put({"error": str(e)})

# layout > instance > mask/path ---> layout > mask/path
def flatten_instances(l_i_mp: DefaultDict[int, DefaultDict[int, List[MaskPath]]]) -> DefaultDict[int, List[MaskPath]]:
  l_mp = defaultdict(list)
  for layout in l_i_mp:
    for instance in l_i_mp[layout]:
      l_mp[layout] += l_i_mp[layout][instance]
  return l_mp

def count_samples(layout_to_samples: Dict[int, List]):
  return list(len(v) for v in layout_to_samples.values())
    
get_frame_id = lambda x: int(os.path.splitext(x)[0])

def tensorize_tokens(tokens: np.ndarray, embed_dim):
  pe = Tensor(get_2d_pos_embed(tokens, embed_dim), requires_grad=False)
  # Throw out last two elements of last axis, which contained x,y-coord data
  # Now layout is only zeroes and ones
  tokens = tokens[:,:,:,0].astype(np.bool)
  tokens = Tensor(tokens, requires_grad=False).unsqueeze(-1).permute(0,3,1,2)
  return tokens, pe

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
  tokens = distance_sort(tokens)
  return tokens

# frequencies (denoms) based on poe-learning-layouts 1d positions embeds for time
# Here we are trying to embed the 2d position of the patch relative to the map entrance
# We want to encode any possible (x,y) position where x and y are any integer
def get_2d_pos_embed(tokens, dim):
  assert dim % 4 == 0
  num_tokens = tokens.shape[0]
  x_coords = tokens[:,0,0,1].reshape(tokens.shape[0], 1)
  y_coords = tokens[:,0,0,2].reshape(tokens.shape[0], 1)
  embeds = np.zeros((num_tokens, dim))
  denoms = np.exp(np.arange(0, dim, 4) / dim * -np.log(10000.0)).reshape(1, dim // 4)
  embeds[:, 0::4] = np.sin(x_coords * denoms) 
  embeds[:, 1::4] = np.cos(x_coords * denoms) 
  embeds[:, 2::4] = np.sin(y_coords * denoms) 
  embeds[:, 3::4] = np.cos(y_coords * denoms) 
  # at this point, dtype is float64
  return embeds.astype(np.float32)

if __name__=="__main__":

  #model = AttentionUNet("AttentionUNet8_8600", depth=3).load()
  #prepare_training_data("data/train", model)
  dl = ViTDataLoader(data_dir="data/train", test_dir="data/test")
  #for _ in range(10):
    #t1 = time.perf_counter()
    #X_steps, Y_steps = dl.get_epoch(min_samples_per_class_per_step=5)
    #print(f"elapsed: {(time.perf_counter() - t1):0.3f}")

  done = 1
