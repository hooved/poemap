# influenced by:
# https://github.com/tinygrad/tinygrad/blob/master/extra/models/vit.py and transformer.py
# https://github.com/kweimann/poe-learning-layouts/blob/main/model.py

from tinygrad import Tensor, dtypes, TinyJit
from tinygrad.nn import Conv2d, Linear, GroupNorm
from tinygrad.nn.state import safe_load, load_state_dict
import numpy as np
from typing import List
from models.UNet import ResBlock

# adapted from tinygrad ViT code
class ViT:
  def __init__(self, model_name, num_classes, max_tokens=128, layers=3, embed_dim=256, num_heads=4):
    self.model_name = model_name
    self.embed_dim = embed_dim
    self.max_tokens = max_tokens
    self.embedding = PatchEmbed()
    self.cls_token = Tensor.scaled_uniform(1, 1, embed_dim)
    self.mask_token = Tensor.zeros(1, embed_dim)
    self.tbs = [
      TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=embed_dim*2,
        prenorm=True, act=lambda x: x.gelu())
      for i in range(layers)]
    self.encoder_norm = (Tensor.scaled_uniform(embed_dim), Tensor.zeros(embed_dim))
    self.head = (Tensor.scaled_uniform(embed_dim, num_classes), Tensor.zeros(num_classes))
    self.jit_inference = None
    self.jit_shape = None

  def prep_tokens(self, x: List[np.ndarray]) -> Tensor:
    # need to use np arrays because tinygrad throws errors on noncontiguous assignments during pos embed calcs
    #class_tokens = self.cls_token.add(Tensor.zeros(len(x),1,1))

    # pad each tensor in x with mask tokens, then batch together
    # todo: batch layouts with same # non-mask tokens? lots of uneven batches grouped together here
    for i, layout in enumerate(x):
      pe = Tensor(get_2d_pos_embed(layout, self.embed_dim), requires_grad=False)
      # Throw out last two elements of last axis, which contained x,y-coord data
      # Now layout is only zeroes and ones
      layout = layout[:,:,:,0].astype(np.bool)
      layout = Tensor(layout, requires_grad=False).unsqueeze(-1).permute(0,3,1,2)
      layout = self.embedding(layout).add(pe)
      if layout.shape[0] < self.max_tokens:
        mask_tokens = self.mask_token.add(Tensor.zeros(self.max_tokens - layout.shape[0], 1))
        x[i] = layout.cat(mask_tokens, dim=0)
      else:
        # Use closest tokens to origin, which more represent training data
        # Current assumption is that these tokens are ordered by euclidean distance from origin,
        #  based on implementation of training_data.tokenize_minimap (with distance_sort at end)
        # TODO: should we randomly sample instead of using tokens closest to origin?
        x[i] = layout[0:self.max_tokens]

    return Tensor.stack(*x)

  def run(self, x: Tensor):
    class_tokens = self.cls_token.add(Tensor.zeros(x.shape[0],1,1))
    x = class_tokens.cat(x, dim=1).sequential(self.tbs)
    x = x.layernorm().linear(*self.encoder_norm)
    x = x[:, 0].linear(*self.head)
    return x
    #return x.realize()

  @TinyJit
  def jit_run(self, x: Tensor):
    return self.run(x).realize()

  def __call__(self, x: List[np.ndarray]) -> Tensor:
    x = self.prep_tokens(x)
    return self.run(x)

  def jit_infer(self, x):
    x = self.prep_tokens(x)
    if self.jit_shape != x.shape:
      self.jit_shape = x.shape
      self.jit_inference = self.jit_run
    return self.jit_inference(x)

  def load(self):
    state_dict = safe_load(f"data/model/{self.model_name}.safetensors")
    load_state_dict(self, state_dict)
    return self

class GlobalAvgPool2d:
  def __init__(self):
    pass

  def __call__(self, x:Tensor) -> Tensor:
    # N, C, H, W  ->  N, C
    x = x.reshape(*x.shape[0:2], x.shape[2] * x.shape[3]) # N, C, H*W
    return x.mean(axis=-1)

class PatchEmbed:
  def __init__(self):
    #self.l1 = Conv2d(1, 64, kernel_size=(16,16), stride=16)
    self.units = [
      Conv2d(1, 64, kernel_size=3, padding=1), 
      ResBlock(64, 64), ResBlock(64, 256),
      GroupNorm(256//16, 256), Tensor.relu, GlobalAvgPool2d(),
      #Conv2d(64, 1, kernel_size=3, padding=1), Tensor.relu,
    ]

  def __call__(self, x:Tensor) -> Tensor:
    # 1 * 32 * 32 --> 256
    #return self.l1(x).relu().flatten(1).dropout(0.3)

    return x.sequential(self.units).dropout(0.3)

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

# copy paste from tinygrad/extra/models/transformer.py
class TransformerBlock:
  def __init__(self, embed_dim, num_heads, ff_dim, prenorm=False, act=lambda x: x.relu(), dropout=0.1):
    assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

    self.num_heads = num_heads
    self.head_size = embed_dim // num_heads
    self.prenorm, self.act = prenorm, act
    self.dropout = dropout

    self.query = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
    self.key = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
    self.value = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))

    self.out = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))

    self.ff1 = (Tensor.scaled_uniform(embed_dim, ff_dim), Tensor.zeros(ff_dim))
    self.ff2 = (Tensor.scaled_uniform(ff_dim, embed_dim), Tensor.zeros(embed_dim))

    self.ln1 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))
    self.ln2 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))

  def attn(self, x):
    # x: (bs, time, embed_dim) -> (bs, time, embed_dim)
    query, key, value = [x.linear(*y).reshape(shape=(x.shape[0], -1, self.num_heads, self.head_size)).transpose(1,2) for y in [self.query, self.key, self.value]]
    attention = Tensor.scaled_dot_product_attention(query, key, value).transpose(1,2)
    return attention.reshape(shape=(x.shape[0], -1, self.num_heads * self.head_size)).linear(*self.out)

  def __call__(self, x):
    if self.prenorm:
      x = x + self.attn(x.layernorm().linear(*self.ln1)).dropout(self.dropout)
      x = x + self.act(x.layernorm().linear(*self.ln2).linear(*self.ff1)).linear(*self.ff2).dropout(self.dropout)
    else:
      x = x + self.attn(x).dropout(self.dropout)
      x = x.layernorm().linear(*self.ln1)
      x = x + self.act(x.linear(*self.ff1)).linear(*self.ff2).dropout(self.dropout)
      x = x.layernorm().linear(*self.ln2)
    return x

if __name__=="__main__":
  from tinygrad.nn.state import safe_load, load_state_dict
  model = ViT(9, max_tokens=128, layers=3, embed_dim=256, num_heads=4)
  model_name = "ViT1"
  state_dict = safe_load(f"data/model/{model_name}.safetensors")
  load_state_dict(model, state_dict)

  x = np.load("data/train/3/2/1.npz")['data']
  pred = model([x])
  print(pred.numpy())
  pred = pred.argmax(axis=1).cast(dtypes.uint8).numpy()
  print(pred)
