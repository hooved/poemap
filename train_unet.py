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

def train(model: Union[UNet, AttentionUNet], patch_size: int=64, batch_size: int=128,
          ga_max_batch: int=256, steps: int=500, lr: float=0.001):

  # accumulate gradients if needed
  batch_size_schedule = [ga_max_batch for _ in range(batch_size // ga_max_batch)] + [batch_size % ga_max_batch]
  batch_size_schedule = [x for x in batch_size_schedule if x > 0]
  
  model.dl.patch_size = (patch_size, patch_size)
  optim = nn.optim.Adam(nn.state.get_parameters(model), lr=lr)

  #@TinyJit
  # GPU memory explodes with TinyJit, currently
  def train_step():
    optim.zero_grad()
    acc_loss = 0
    for ga_bs in batch_size_schedule:
      X, Y = model.dl.get_batch(ga_bs)
      pred = model.__call__(X)
      s = pred.shape
      loss = pred.permute(0,2,3,1).reshape(-1, s[1]).cross_entropy(Y.reshape(-1)) * ga_bs / batch_size
      loss.backward()
      acc_loss += loss.item()
      loss = multiclass_dice_loss(pred, Y) * ga_bs / batch_size
      loss.backward()
      if len(batch_size_schedule) > 1:
        cleanup_grads(optim)
      acc_loss += loss.item()
    optim.step()
    return acc_loss

  def cleanup_grads(optim: nn.optim.LAMB):
    # hack to accumulate gradients without memory leak
    for t in optim.params:
      if hasattr(t, 'grad') and t.grad is not None:
        t_np = t.grad.numpy()
        if hasattr(t.grad, 'lazydata'):
          if hasattr(t.grad.lazydata, 'buffer'):
            del t.grad.lazydata.buffer
          del t.grad.lazydata
        del t.grad
        t.grad = Tensor(t_np, requires_grad=False)

  @TinyJit
  def eval_step(ga_bs):
    X_test, Y_test = model.dl.get_batch(ga_bs)
    acc = (model.__call__(X_test).argmax(axis=1) == Y_test).mean()
    return acc.realize()

  for i in range(steps):
    Tensor.training = True
    #loss = train_step().item()
    acc = 0
    loss = train_step()
    if i%5 == 0:
      print(f"step {i:4d}, loss {loss:.4f}")
    if i%15 == 0 and i != 0:
        safe_save(get_state_dict(model), f"data/model/{model.model_name}_{i}.safetensors")
      #Tensor.training = False
      #acc = eval_step().item()
      #print(f"step {i:4d}, loss {loss:.4f}, acc {acc*100.:.2f}%")
      #print(f"step {i:4d}, loss {loss:.4f}")
    if WANDB:
      wandb.log({"step": i, "loss": loss, "accuracy": acc})

if __name__=="__main__":
  config = {}
  patch_size = config["patch_size"] = 32
  num_steps = config["num_steps"] = 150
  batch_size = config["batch_size"] = 256
  lr = config["learning_rate"] = 0.001
  model_name = config["model_name"] = "UNet2"

  model = UNet(model_name)
  #model = AttentionUNet(model_name, depth=1)

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

  try:
    train(model, patch_size=patch_size, steps=num_steps, batch_size=batch_size, lr=lr)
  finally:
    # save model if we ctrl+c
    safe_save(get_state_dict(model), f"data/model/{model.model_name}.safetensors")