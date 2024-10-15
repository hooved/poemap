from comms import receive_int, receive_array, send_array, ClientHeader
import numpy as np
from PIL import Image
import asyncio

async def minimap_to_layout(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
  print("connection opened")
  while True:
    header = await receive_int(reader)
    header = ClientHeader(header)

    if header == ClientHeader.PROCESS_MINIMAP:
      minimap = await receive_array(reader)
      print(f"received minimap, shape = {minimap.shape}")
      Image.fromarray(minimap).save("minimap.png")
      response = 7
      writer.write(response.to_bytes(4, byteorder="big"))
      await writer.drain()

    elif header == ClientHeader.CLOSE_CONNECTION:
      print("closing connection")
      writer.close()
      await writer.wait_closed()
      break

async def run_server(hostname, port: int):
  print(f"starting server at {hostname}:{port}")
  server = await asyncio.start_server(minimap_to_layout, hostname, port)
  async with server:
    await server.serve_forever()

if __name__=="__main__":
   asyncio.run(run_server('localhost', 50000))