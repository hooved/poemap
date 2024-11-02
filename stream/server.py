from comms import receive_int, receive_array, send_array, ClientHeader
import numpy as np
from PIL import Image
import asyncio, os
from models import AttentionUNet, ViT
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad import dtypes, Tensor, TinyJit
from training_data import extract_map_features, get_patches, get_tokens, tokenize_minimap
from functools import partial
import time

async def minimap_to_layout(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, models: dict):
  print("connection opened")
  timestamp = 0
  while True:
    header = await receive_int(reader)
    header = ClientHeader(header)

    if header == ClientHeader.PROCESS_MINIMAP:
      try:
        minimap = await receive_array(reader, dtype=np.uint8)
        origin = await receive_array(reader, dtype=np.uint32)
        tokens, mask = tokenize_minimap(minimap, origin, models["UNet"])
        # if mask is blank, will get error later on in ViT
        if np.any(mask != 0):
          if os.getenv("COLLECT"):
            save_dir = os.path.join("data", "train", "collect")
            os.makedirs(save_dir, exist_ok=True)
            Image.fromarray(minimap).save(os.path.join(save_dir, f"{timestamp}.png"))
            np.savez_compressed(os.path.join(save_dir, f"{timestamp}_origin.npz"), data=origin)
            Image.fromarray(mask * 255, mode="L").save(os.path.join(save_dir, f"{timestamp}_mask.png"))
            np.savez_compressed(os.path.join(save_dir, f"{timestamp}.npz"), data=tokens)
          timestamp += 1
          print(tokens.shape)
          layout_id = models["ViT"]([tokens])[0].argmax().cast(dtypes.uint8).item()
          writer.write(layout_id.to_bytes(4, byteorder="big"))
          await writer.drain()
        else:
          error_val = 9999
          writer.write(error_val.to_bytes(4, byteorder="big"))
          await writer.drain()
      except:
        Image.fromarray(minimap).save("failed_minimap.png")
        Image.fromarray(mask * 255, mode="L").save("failed_mask.png")
        assert False

    elif header == ClientHeader.SAVE_FRAME:
      frame = await receive_array(reader, dtype=np.uint8)
      save_dir = os.path.join("data", "background", "collect")
      os.makedirs(save_dir, exist_ok=True)
      Image.fromarray(frame).save(os.path.join(save_dir, f"{timestamp}.png"))
      timestamp += 1
      response = 42
      writer.write(response.to_bytes(4, byteorder="big"))
      await writer.drain()

    elif header == ClientHeader.CLOSE_CONNECTION:
      print("closing connection")
      writer.close()
      await writer.wait_closed()
      break

async def run_server(hostname, port: int):
  models = {
    "UNet": AttentionUNet("AttentionUNet8_8600", depth=3).load(),
    "ViT": ViT("ViT3_2399", num_classes=9, max_tokens=128, layers=3, embed_dim=256, num_heads=4).load(),
    #"ViT": ViT("ViT3_799", num_classes=9, max_tokens=128, layers=3, embed_dim=256, num_heads=4).load(),
  }
  warmup = models["UNet"].batch_inference(np.random.randint(0, 256, size=(32*10, 32*10, 3), dtype=np.uint8), chunk_size=32)
  print(f"UNet warmup.shape: {warmup.shape}")
  warmup = models["ViT"].jit_infer([np.random.randint(0, 2, size=(64,32,32,3), dtype=np.int64)])
  warmup = warmup.argmax(axis=1).cast(dtypes.uint8).numpy()[0]
  print(f"ViT warmup: {warmup}")

  print(f"starting server at {hostname}:{port}")
  server = await asyncio.start_server(partial(minimap_to_layout, models=models), hostname, port)
  async with server:
    await server.serve_forever()

if __name__=="__main__":
  asyncio.run(run_server('localhost', 50000))