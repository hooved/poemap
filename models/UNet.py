from tinygrad import Tensor, TinyJit, nn
from tinygrad.nn import Conv2d, ConvTranspose2d, BatchNorm2d
from tinygrad.dtype import dtypes
from tinygrad.nn.state import safe_load, safe_save, get_state_dict, load_state_dict
import numpy as np
from dataloader import DataLoader

"""
Adapted from:
https://github.com/milesial/Pytorch-UNet
https://github.com/tinygrad/tinygrad/examples/stable_diffusion.py
https://docs.tinygrad.org/mnist/
https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py
"""

def doubleconv(in_chan, out_chan):
  return [Conv2d(in_chan, out_chan, kernel_size=3, padding=1), BatchNorm2d(out_chan), Tensor.relu,
    Conv2d(out_chan, out_chan, kernel_size=3, padding=1), BatchNorm2d(out_chan), Tensor.relu]

class UNet:
  def __init__(self, model_name, in_chan=3, mid_chan=64, out_chan=2, depth=2):
    self.model_name = model_name
    self.dl = DataLoader(
      #image_dir="data/auto_crop",
      #mask_dir="data/mask",
      image_dir="data/auto_crop_50",
      mask_dir="data/mask_50",
    )
    self.save_intermediates = [
      doubleconv(in_chan, mid_chan), 
    ]
    self.consume_intermediates = [
      [*doubleconv(mid_chan * 2, mid_chan), Conv2d(mid_chan, out_chan, kernel_size=1)],
    ]

    for i in range(depth-1):
      self.save_intermediates += [[Tensor.max_pool2d, *doubleconv(mid_chan * 2**i, mid_chan * 2**(i+1))]]
      self.consume_intermediates = [
          [*doubleconv(mid_chan * 2**(i+2), mid_chan * 2**(i+1)), 
          ConvTranspose2d(mid_chan * 2**(i+1), mid_chan * 2**i, kernel_size=2, stride=2)]
        ] + self.consume_intermediates

    self.middle = [
      Tensor.max_pool2d, *doubleconv(mid_chan * 2**(depth-1), mid_chan * 2**depth),
      ConvTranspose2d(mid_chan * 2**depth, mid_chan * 2**(depth-1), kernel_size=2, stride=2),
    ]

  def __call__(self, x):
    intermediates = []
    for b in self.save_intermediates:
      for bb in b:
        x = bb(x)
      intermediates.append(x)
    for bb in self.middle:
      x = bb(x)
    for b in self.consume_intermediates:
      x = intermediates.pop().cat(x, dim=1)
      for bb in b:
        x = bb(x)
    return x

  def load(self):
    state_dict = safe_load(f"data/model/{self.model_name}.safetensors")
    load_state_dict(self, state_dict)
    return self

  def batch_inference(self, whole, chunk_size=64, batch_size=64):
    original_shape = whole.shape
    chunks = self.dl.get_model_input_chunks(whole, chunk_size=chunk_size)
    def run(self, x):
      Tensor.training = False
      return self.__call__(x).argmax(axis=1, keepdim=True).cast(dtypes.uint8).permute(0,2,3,1).numpy()
    jit_run = TinyJit(run)
    # Inference on the whole image takes too much GPU memory, so we run inference on subsets
    result = np.empty((0, chunk_size, chunk_size, 1), dtype=np.uint8)
    for i in range(0, chunks.shape[0], batch_size):
      model_input = Tensor(chunks[i:i + batch_size])
      # TinyJit throws exception when the tensor shape changes
      if i + batch_size <= chunks.shape[0]:
        model_output = jit_run(self, model_input)
      else:
        model_output = run(self, model_input)
      result = np.concatenate((result, model_output), axis=0)

    result = self.dl.synthesize_image_from_chunks(result, (*original_shape[0:2], 1)).squeeze(-1)
    return result

class AttentionBlock:
  def __init__(self, g_chan, l_chan, int_chan):
    self.W_g = [Conv2d(g_chan, int_chan, 1, stride=1), BatchNorm2d(int_chan)]
    self.W_x = [Conv2d(l_chan, int_chan, 1, stride=1), BatchNorm2d(int_chan)]
    self.psi = [Conv2d(int_chan, 1, 1, stride=1), BatchNorm2d(1), Tensor.sigmoid]

  def __call__(self, g: Tensor, x: Tensor):
    psi = Tensor.relu(g.sequential(self.W_g) + x.sequential(self.W_x))
    psi = psi.sequential(self.psi)
    return x * psi

class AttentionUNet(UNet):
  def __init__(self, model_name, in_chan=3, mid_chan=64, out_chan=2, depth=2):
    super().__init__(model_name, in_chan=in_chan, mid_chan=mid_chan, out_chan=out_chan, depth=depth)

    self.attention_blocks = []
    for i in range(depth):
      input_chan = mid_chan * 2**i
      self.attention_blocks = [AttentionBlock(input_chan, input_chan, input_chan//2)] + self.attention_blocks

  def __call__(self, x):
    intermediates = []
    for b in self.save_intermediates:
      for bb in b:
        x = bb(x)
      intermediates.append(x)
    for bb in self.middle:
      x = bb(x)
    for i, b in enumerate(self.consume_intermediates):
      intermediate = intermediates.pop()
      intermediate = self.attention_blocks[i](x, intermediate)
      x = intermediate.cat(x, dim=1)
      for bb in b:
        x = bb(x)
    return x