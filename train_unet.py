import os, sys
from models import UNet, AttentionUNet
from typing import Tuple, Optional, Union
from tinygrad import Tensor, TinyJit, nn
from tinygrad.nn.state import safe_save, get_state_dict

if WANDB := os.getenv("WANDB"):
  import wandb

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

def train(model, patch_size: Optional[Tuple[int]]=(64,64), batch_size: Optional[int]=128,
          steps: Optional[int]=500, lr=0.001):
  model.dl.patch_size = patch_size
  optim = nn.optim.Adam(nn.state.get_parameters(model), lr=lr)
  def step():
    Tensor.training = True 
    X, Y = model.dl.get_batch(batch_size)
    optim.zero_grad()
    pred = model.__call__(X)
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
    loss = jit_step().item()
    acc = None
    if step%5 == 0:
      Tensor.training = False
      #X_test, Y_test = model.dl.get_batch(batch_size, test=True)
      X_test, Y_test = model.dl.get_batch(batch_size)
      acc = (model.__call__(X_test).argmax(axis=1) == Y_test).mean().item()
      print(f"step {step:4d}, loss {loss:.2f}, acc {acc*100.:.2f}%")
    if WANDB:
      wandb.log({"step": step, "loss": loss, "accuracy": acc})

  safe_save(get_state_dict(model), f"data/model/{model.model_name}.safetensors")

if __name__=="__main__":
  model_name = "UNet1"
  model = UNet(model_name)
  config = {}
  patch_size = config["patch_size"] = 64
  num_steps = config["num_steps"] = 300
  batch_size = config["batch_size"] = 128
  lr = config["learning_rate"] = 0.001
  config = {
    "patch_size": patch_size,
    "num_steps": num_steps,
    "batch_size": batch_size,
    "learning_rate": lr,
  }

  if WANDB:
    wandb.init(project="poemap", config=config)

  train(model, patch_size=patch_size, steps=num_steps, batch_size=batch_size, lr=lr)