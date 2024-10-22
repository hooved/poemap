from tinygrad import Tensor
import numpy as np
import os, random

class ImageMaskPair:
  def __init__(self, image_path, mask_path):
    self.image_path = image_path
    self.mask_path = mask_path

  def load_image(self):
    return np.load(self.image_path)['data']

  def load_mask(self):
    return np.load(self.mask_path)['data']

class DataLoader:
  def __init__(self, image_dir, mask_dir, patch_size=(64, 64), normalize=True, 
         flip_prob=0.5, rotate_prob=0.5, noise_prob=0,):
    self.image_dir = image_dir
    self.mask_dir = mask_dir
    self.patch_size = patch_size
    self.normalize = normalize
    self.flip_prob = flip_prob
    self.rotate_prob = rotate_prob
    self.noise_prob = noise_prob
    self.image_mask_pairs = self.get_image_mask_pairs()

  def get_image_mask_pairs(self):
    ret = []
    for subdir in os.listdir(self.image_dir):
      for file in os.listdir(os.path.join(self.image_dir, subdir)):
        im_file = os.path.join(self.image_dir, subdir, file)
        mask_file = os.path.join(self.mask_dir, subdir + ".npz")
        ret.append(ImageMaskPair(im_file, mask_file))

    ret = sorted(ret, key = lambda x: x.mask_path)
    return ret

  def get_batch(self, batch_size):
    # Randomly distribute samples across images
    shares = np.random.dirichlet(np.ones(len(self.image_mask_pairs)), size=1)[0]
    result = np.round(shares * batch_size).astype(int)
    # Adjust to ensure sum is exactly batch_size
    diff = batch_size - result.sum()
    result[np.argmax(result)] += diff

    image_patches, mask_patches = [], []
    mask_cache = {}
    for i, num_samples in enumerate(result):
      imp = self.image_mask_pairs[i]
      if mask_cache.get(imp.mask_path) is None:
        # We sorted image_mask_pairs by mask, so we don't need to cache previous masks
        mask_cache = {imp.mask_path: imp.load_mask()}
      mask = mask_cache[imp.mask_path]
      image = imp.load_image()
      image = self._normalize(image) if self.normalize else image
      for _ in range(num_samples):
        ip, mp = self._random_crop(image, mask)
        ip, mp = self._apply_augmentations(ip, mp)
        image_patches.append(ip)
        mask_patches.append(mp)
    image_patches = Tensor(image_patches).permute(0,3,1,2)
    return image_patches, Tensor(mask_patches)

  def _apply_augmentations(self, image, mask):
    if random.random() < self.flip_prob:
      image, mask = self._random_flip(image, mask)
    if random.random() < self.rotate_prob:
      image, mask = self._random_rotate(image, mask)
    if random.random() < self.noise_prob:
      image = self._random_noise(image)  # Apply noise only to the image, not the mask
    return image, mask

  def _random_crop(self, image, mask):
    h, w = image.shape[:2]
    new_h, new_w = self.patch_size

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    image_patch = image[top:top+new_h, left:left+new_w]
    mask_patch = mask[top:top+new_h, left:left+new_w]

    return image_patch, mask_patch

  def _random_flip(self, image, mask):
    return np.fliplr(image), np.fliplr(mask)

  def _random_rotate(self, image, mask):
    k = random.choice([1, 2, 3])  # 90, 180, or 270 degrees
    return np.rot90(image, k), np.rot90(mask, k)

  def _random_noise(self, image):
    noise = np.random.normal(0, 0.05, image.shape)
    return np.clip(image + noise, 0, 1)

  def _normalize(self, image):
    #return (image - np.mean(image)) / np.std(image)
    normalized = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[2]):
      channel = image[:,:,i]
      mean = np.mean(channel)
      std = np.std(channel)
      normalized[:,:,i] = (channel - mean) / (std + 1e-8)  # adding small epsilon to avoid division by zero
    return normalized

  def prep(self, image):
    return Tensor(self._normalize(image), requires_grad=False).permute(2,0,1).unsqueeze(0)

  def split_image_into_chunks(self, image, chunk_size=64):
    height, width, channels = image.shape
    chunks_h = height // chunk_size
    chunks_w = width // chunk_size
    reshaped = image.reshape(chunks_h, chunk_size, chunks_w, chunk_size, channels)
    return reshaped.transpose(0, 2, 1, 3, 4).reshape(-1, chunk_size, chunk_size, channels)
    
  def get_model_input_chunks(self, image, chunk_size=64):
    chunks = self.split_image_into_chunks(self._normalize(image), chunk_size)
    return chunks.transpose(0,3,1,2)
  
  def synthesize_image_from_chunks(self, chunks, original_shape):
    height, width, channels = original_shape
    chunk_size = chunks.shape[1]  # Assuming chunks are square
    chunks_h = height // chunk_size
    chunks_w = width // chunk_size
    reshaped = chunks.reshape(chunks_h, chunks_w, chunk_size, chunk_size, channels)
    transposed = reshaped.transpose(0, 2, 1, 3, 4)
    return transposed.reshape(height, width, channels)