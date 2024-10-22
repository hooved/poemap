import os, sys, math
from models import UNet, AttentionUNet
from typing import Tuple, Optional, Union
from tinygrad import Tensor, TinyJit, nn
from tinygrad.nn.state import safe_save, get_state_dict

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

def train(model, patch_size: Optional[int]=64, batch_size: Optional[int]=128,
          ga_max_batch: int=128, steps: Optional[int]=500, lr=0.001):

  # accumulate gradients if needed
  batch_size_schedule = [ga_max_batch for _ in range(batch_size // ga_max_batch)] + [batch_size % ga_max_batch]
  
  model.dl.patch_size = (patch_size, patch_size)
  optim = nn.optim.Adam(nn.state.get_parameters(model), lr=lr)

  @TinyJit
  def train_step():
    optim.zero_grad()
    acc_loss = 0

    for ga_bs in batch_size_schedule:
      X, Y = model.dl.get_batch(ga_bs)
      pred = model.__call__(X)
      s = pred.shape
      loss = pred.permute(0,2,3,1).reshape(-1, s[1]).cross_entropy(Y.reshape(-1))
      weight = 1
      #combined_loss = (loss + weight * multiclass_dice_loss(pred, Y)).backward()
      combined_loss = (loss + weight * multiclass_dice_loss(pred, Y)) * ga_bs / batch_size
      combined_loss.backward()
      acc_loss = acc_loss + combined_loss

    optim.step()
    return acc_loss.realize()

  @TinyJit
  def eval_step():
    acc = 0
    for ga_bs in batch_size_schedule:
      X_test, Y_test = model.dl.get_batch(ga_bs)
      accumulate = (model.__call__(X_test).argmax(axis=1) == Y_test).mean()
      accumulate = accumulate * ga_bs / batch_size
      acc = acc + accumulate
    return acc.realize()

  for i in range(steps):
    Tensor.training = True
    loss = train_step().item()
    acc = None
    if i%5 == 0:
      Tensor.training = False
      acc = eval_step().item()
      print(f"step {i:4d}, loss {loss:.4f}, acc {acc*100.:.2f}%")
    if WANDB:
      wandb.log({"step": i, "loss": loss, "accuracy": acc})

  safe_save(get_state_dict(model), f"data/model/{model.model_name}.safetensors")

if __name__=="__main__":
  config = {}
  patch_size = config["patch_size"] = 64
  num_steps = config["num_steps"] = 600
  batch_size = config["batch_size"] = 512
  lr = config["learning_rate"] = 0.01
  model_name = config["model_name"] = "UNet5"

  model = UNet(model_name)

  if WANDB := os.getenv("WANDB"):
    os.environ['WANDB_HOST'] = 'poemap'
    import wandb
    settings={
        "_disable_stats": True,  # disable collecting system metrics
        "_disable_meta": True,  # disable collecting system metadata (including hardware info)
        "_disable_machine_info": True,  # disable collecting system metadata (including hardware info)
        "console": "off",  # disable capturing stdout/stderr
    }
    wandb.init(project="poemap", settings=settings, config=config)

  train(model, patch_size=patch_size, steps=num_steps, batch_size=batch_size, lr=lr)