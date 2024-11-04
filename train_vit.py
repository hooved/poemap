from models import ViT
import numpy as np
from tinygrad import Tensor, nn, dtypes, TinyJit
from tinygrad.nn.state import safe_load, safe_save, get_state_dict, load_state_dict
from training_data import ViTDataLoader

if __name__=="__main__":
  dl = ViTDataLoader(data_dir="data/train")

  model_name = "ViT5"
  #model = ViT(model_name, 9, max_tokens=128, layers=3, embed_dim=1024, num_heads=4)
  model = ViT(model_name, 9, max_tokens=128, layers=3, embed_dim=256, num_heads=4)
  #tokens = np.load("data/train/1/0/0.npz")['data']
  optim = nn.optim.Adam(nn.state.get_parameters(model))

  def step(X, Y):
    Tensor.training = True
    optim.zero_grad()
    #pred = model(X)
    #pred = pred.argmax(axis=1).cast(dtypes.uint8).numpy()
    loss = model(X).cross_entropy(Tensor(Y)).backward()
    optim.step()
    return loss.realize()
  jit_step = TinyJit(step)

  num_steps = 20000

  try:
    last_saved_loss = float("inf")
    elapsed = 0
    for step_id in range(num_steps):
      #for i, (X, Y) in enumerate(dl.get_training_data(model.max_tokens)):
      #X, Y = dl.get_training_data(max_tokens=model.max_tokens)

      # For fast testing
      if step_id >= 20:
        break
      X = [np.random.randint(0, 2, size=(8,32,32,3), dtype=np.int64)]
      Y = [1]

      loss = jit_step(X, Y).item()
      elapsed += 1
      if step_id % 5 == 0:
        print(f"step: {step_id:5d}, loss: {loss:.7f}")

      if elapsed >= 200 and loss < last_saved_loss:
        elapsed = 0
        last_saved_loss = loss
        safe_save(get_state_dict(model), f"data/model/{model.model_name}_{step_id}.safetensors")
  finally:
    safe_save(get_state_dict(model), f"data/model/{model_name}.safetensors")