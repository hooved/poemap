from models import ViT
import numpy as np
from tinygrad import Tensor, nn, dtypes, TinyJit
from training_data import ViTDataLoader

if __name__=="__main__":
  dl = ViTDataLoader(data_dir="data/train")
  ctp = dl.class_to_paths

  model = ViT(9, max_tokens=128, layers=3, embed_dim=256, num_heads=4)
  #tokens = np.load("data/train/1/0/0.npz")['data']
  optim = nn.optim.Adam(nn.state.get_parameters(model))

  def step():
    Tensor.training = True
    optim.zero_grad()
    tokens = np.load("data/train/1/0/0.npz")['data']
    pred = model([tokens])
    truth = Tensor([1])
    loss = pred.cross_entropy(truth).backward()
    optim.step()
    pred = pred.argmax(axis=1).cast(dtypes.uint8).numpy()
    print(pred)
    return loss
  jit_step = TinyJit(step)

  for step_num in range(1):
    loss = jit_step()
    print(f"step: {step_num:4d}, loss: {loss.item():.2f}")

  done = 1