from models import ViT
from tinygrad import Tensor, nn, TinyJit
from tinygrad.nn.state import safe_save, get_state_dict
from training_data import ViTDataLoader
from typing import List, Tuple

if __name__=="__main__":
  dl = ViTDataLoader(data_dir="data/train", test_dir="data/test")

  model_name = "ViT6"
  model = ViT(model_name, 9, max_tokens=128, layers=3, embed_dim=256, num_heads=4)

  X_test, Y_test = dl.get_test_data()
  
  optim = nn.optim.Adam(nn.state.get_parameters(model))

  def step(X: Tensor, Y: Tensor, batch_pad_mask: Tensor):
    loss = model.run(X).cross_entropy(Y, reduction="none")
    # zero out losses from pad samples
    loss = loss * batch_pad_mask
    loss = loss.sum() / batch_pad_mask.sum()
    loss.backward()
    optim.step()
    return loss.realize()
  jit_step = TinyJit(step)

  def eval_step(X: List[Tuple[Tensor]], Y: Tensor):
    acc = (model(X).argmax(axis=1) == Y).mean()
    return acc.realize()
  jit_eval = TinyJit(eval_step)

  num_epochs = 1000
  try:
    last_saved_loss = float("inf")
    elapsed = 0

    for epoch in range(num_epochs):
      for step_id, (X, Y, batch_pad_mask) in enumerate(dl.get_epoch(min_samples_per_class_per_step=5)):

        Tensor.training = True
        optim.zero_grad()
        X = model.prep_tokens(X)
        loss = jit_step(X, Y, batch_pad_mask).item()
        elapsed += 1

        if step_id % 5 == 0:
          Tensor.training = False
          acc = jit_eval(X_test, Y_test).item()
          print(f"epoch: {epoch:4d}, step: {step_id:4d}, loss: {loss:.7f}, acc: {acc:0.4f}")

        if elapsed >= 20 and loss < last_saved_loss:
          elapsed = 0
          last_saved_loss = loss
          safe_save(get_state_dict(model), f"data/model/{model.model_name}_{epoch}_{step_id}.safetensors")
  finally:
    safe_save(get_state_dict(model), f"data/model/{model_name}.safetensors")
    print("training done")