from comms import receive_int, receive_array, send_array, ClientHeader
import numpy as np
from PIL import Image
import asyncio, os
from models import AttentionUNet, ViT
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad import dtypes, Tensor, TinyJit
from training_data import extract_map_features, get_patches, get_tokens
from functools import partial

async def minimap_to_layout(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, models: dict):
  print("connection opened")
  timestamp = 0
  while True:
    header = await receive_int(reader)
    header = ClientHeader(header)

    if header == ClientHeader.PROCESS_MINIMAP:
      minimap = await receive_array(reader, dtype=np.uint8)
      origin = await receive_array(reader, dtype=np.uint32)
      origin = tuple(int(x) for x in origin)
      mask, origin = extract_map_features(minimap, origin, models["UNet"])
      tokens = [get_tokens(get_patches(mask, origin))]
      if os.getenv("COLLECT"):
        os.makedirs("data/train/collect", exist_ok=True)
        Image.fromarray(minimap).save(f"data/train/collect/{timestamp}.png")
        np.savez_compressed(f"data/train/collect/{timestamp}.npz")
      timestamp += 1
      layout_id = models["ViT"](tokens)[0].argmax().cast(dtypes.uint8).item()
      writer.write(layout_id.to_bytes(4, byteorder="big"))
      await writer.drain()

    elif header == ClientHeader.CLOSE_CONNECTION:
      print("closing connection")
      writer.close()
      await writer.wait_closed()
      break

async def run_server(hostname, port: int):
  models = {
    "UNet": AttentionUNet("AttentionUNet_4").load(),
    "ViT": ViT("ViT1", num_classes=9, max_tokens=128, layers=3, embed_dim=256, num_heads=4).load(),
  }
  warmup = models["UNet"].batch_inference(np.random.randint(0, 256, size=(32*10, 32*10, 3), dtype=np.uint8), chunk_size=32)
  print(f"UNet warmup.shape: {warmup.shape}")
  warmup = models["ViT"]([np.random.randint(0, 2, size=(64,32,32,3), dtype=np.uint8)])
  warmup = warmup.argmax(axis=1).cast(dtypes.uint8).numpy()[0]
  print(f"ViT warmup: {warmup}")

  print(f"starting server at {hostname}:{port}")
  server = await asyncio.start_server(partial(minimap_to_layout, models=models), hostname, port)
  async with server:
    await server.serve_forever()

if __name__=="__main__":
  asyncio.run(run_server('localhost', 50000))