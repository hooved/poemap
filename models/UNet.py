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
https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py
"""

def doubleconv(in_chan, out_chan):
  return [Conv2d(in_chan, out_chan, kernel_size=3, padding=1), BatchNorm2d(out_chan), Tensor.relu,
    Conv2d(out_chan, out_chan, kernel_size=3, padding=1), BatchNorm2d(out_chan), Tensor.relu]

def multiclass_dice_loss(preds, targets, smooth=1e-6):
    """
    Args:
        preds (Tensor): Predicted logits with shape [batch, num_classes, H, W].
        targets (Tensor): Ground truth labels with shape [batch, H, W].
    Returns:
        Tensor: Dice Loss.
    """
    num_classes = preds.shape[1]
    preds = preds.softmax(axis=1)
    targets_one_hot = targets.one_hot(num_classes=num_classes)  # Shape: [batch, H, W, num_classes]
    targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # Shape: [batch, num_classes, H, W]
    preds_flat = preds.view(preds.shape[0], preds.shape[1], -1)
    targets_flat = targets_one_hot.view(targets_one_hot.shape[0], targets_one_hot.shape[1], -1)

    intersection = (preds_flat * targets_flat).sum(axis=2)
    numerator = 2.0 * intersection + smooth
    denominator = preds_flat.sum(axis=2) + targets_flat.sum(axis=2) + smooth
    dice_score = numerator / denominator
    dice_loss = 1 - dice_score.mean()
    return dice_loss

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

  def train(self, patch_size: Optional[Tuple[int]]=(64,64), batch_size: Optional[int]=128,
            steps: Optional[int]=500):
    self.dl.patch_size = patch_size
    optim = nn.optim.Adam(nn.state.get_parameters(self))
    def step():
      Tensor.training = True 
      X, Y = self.dl.get_batch(batch_size)
      optim.zero_grad()
      pred = self.__call__(X)
      s = pred.shape

      # uncomment this block to incorporate dice loss
      loss = pred.permute(0,2,3,1).reshape(-1, s[1]).cross_entropy(Y.reshape(-1))
      weight = 1
      combined_loss = (loss + weight * multiclass_dice_loss(pred, Y)).backward()
      optim.step()
      return combined_loss

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