from models import ViT
import numpy as np
from tinygrad import Tensor, nn, dtypes, TinyJit
from tinygrad.nn.state import safe_load, safe_save, get_state_dict, load_state_dict
from training_data import ViTDataLoader

if __name__=="__main__":
  dl = ViTDataLoader(data_dir="data/train")

  model = ViT(9, max_tokens=128, layers=3, embed_dim=256, num_heads=4)
  model_name = "ViT2"
  #tokens = np.load("data/train/1/0/0.npz")['data']
  optim = nn.optim.Adam(nn.state.get_parameters(model))

  def step(X, Y):
    Tensor.training = True
    optim.zero_grad()
    #pred = model(X)
    #pred = pred.argmax(axis=1).cast(dtypes.uint8).numpy()
    loss = model(X).cross_entropy(Tensor(Y)).backward()
    optim.step()
    return loss
  jit_step = TinyJit(step)

  num_epochs = 200
  for epoch in range(num_epochs):
    for i, (X, Y) in enumerate(dl.get_training_data(model.max_tokens)):
      loss = jit_step(X, Y)
      print(f"epoch: {epoch:4d}, step: {i:4d}, loss: {loss.item():.2f}")

  safe_save(get_state_dict(model), f"data/model/{model_name}.safetensors")

  done = 1