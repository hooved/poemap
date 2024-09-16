from tinygrad import Tensor, TinyJit, nn
from tinygrad.nn import Conv2d, ConvTranspose2d, BatchNorm2d
from tinygrad.dtype import dtypes
from tinygrad.nn.state import safe_load, safe_save, get_state_dict, load_state_dict
import numpy as np
from dataloader import DataLoader
from typing import Tuple, Optional, Union

"""
Adapted from:
https://github.com/milesial/Pytorch-UNet
https://github.com/tinygrad/tinygrad/examples/stable_diffusion.py
https://docs.tinygrad.org/mnist/
"""

def doubleconv(in_chan, out_chan):
  return [Conv2d(in_chan, out_chan, kernel_size=3, padding=1), BatchNorm2d(out_chan), Tensor.relu,
    Conv2d(out_chan, out_chan, kernel_size=3, padding=1), BatchNorm2d(out_chan), Tensor.relu]

class UNet:
  def __init__(self, model_name):
    self.model_name = model_name
    self.dl = DataLoader(
      image_dir="data/auto_crop",
      mask_dir="data/mask",
    )
    self.save_intermediates = [
      doubleconv(3, 64), 
      [Tensor.max_pool2d, *doubleconv(64, 128)],
    ]
    self.middle = [
      Tensor.max_pool2d, *doubleconv(128, 256),
      ConvTranspose2d(256, 128, kernel_size=2, stride=2),
    ]
    self.consume_intermediates = [
      [*doubleconv(256, 128), ConvTranspose2d(128, 64, kernel_size=2, stride=2)],
      [*doubleconv(128, 64), Conv2d(64, 2, kernel_size=1)],
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

  @classmethod
  def load(cls, model_name):
    state_dict = safe_load(f"data/model/{model_name}.safetensors")
    model = cls(model_name)
    load_state_dict(model, state_dict)
    return model

  def train(self, patch_size: Optional[Tuple[int]]=(64,64), batch_size: Optional[int]=64,
            steps: Optional[int]=200):
    self.dl.patch_size = patch_size
    optim = nn.optim.Adam(nn.state.get_parameters(self))
    def step():
      Tensor.training = True 
      X, Y = self.dl.get_batch(batch_size)
      optim.zero_grad()
      pred = self.__call__(X)
      s = pred.shape
      # Need to flatten for cross_entropy to work
      loss = pred.permute(0,2,3,1).reshape(-1, s[1]).cross_entropy(Y.reshape(-1)).backward()
      optim.step()
      return loss
    jit_step = TinyJit(step)

    for step in range(steps):
      loss = jit_step()
      if step%5 == 0:
        Tensor.training = False
        X_test, Y_test = self.dl.get_batch(batch_size)
        acc = (self.__call__(X_test).argmax(axis=1) == Y_test).mean().item()
        print(f"step {step:4d}, loss {loss.item():.2f}, acc {acc*100.:.2f}%")

    safe_save(get_state_dict(self), f"data/model/{self.model_name}.safetensors")

  def batch_inference(self, whole, chunk_size=64, batch_size=64):
    original_shape = whole.shape
    chunks = self.dl.get_model_input_chunks(whole, chunk_size=chunk_size)
    def run(self, x):
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

class UNet2(UNet):
  def __init__(self):
    super().__init__()
    self.save_intermediates += [self.middle[0:1 + 6]]
    self.middle = [
      Tensor.max_pool2d, *doubleconv(256, 512),
      ConvTranspose2d(512, 256, kernel_size=2, stride=2),
    ]
    self.consume_intermediates = [
      [*doubleconv(512, 256), ConvTranspose2d(256, 128, kernel_size=2, stride=2)],
    ] + self.consume_intermediates
