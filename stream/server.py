from comms import receive_array, send_array
import numpy as np
from PIL import Image
import asyncio

async def minimap_to_layout(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
  print("connection opened")
  while True:
    minimap = await receive_array(reader)

    if minimap.shape == (42,):
      print("closing connection")
      writer.close()
      await writer.wait_closed()
      break

    print(f"received minimap, shape = {minimap.shape}")
    print(minimap)
    #minimap = Image.fromarray(minimap)
    #minimap.save("minimap.png")

    response = np.array([1,2,3], dtype=np.uint8)
    await send_array(writer, response)
    #writer.close()
    #await writer.wait_closed()

async def run_server(hostname, port: int):
  print(f"starting server at {hostname}:{port}")
  server = await asyncio.start_server(minimap_to_layout, hostname, port)
  async with server:
    await server.serve_forever()

if __name__=="__main__":
   asyncio.run(run_server('localhost', 50000))