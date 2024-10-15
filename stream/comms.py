import numpy as np
import functools, operator, asyncio

async def receive_exact(reader: asyncio.StreamReader, n: int) -> bytes:
  data = b''
  while len(data) < n:
    packet = await reader.read(n - len(data))
    if not packet:
      return None
    data += packet
  return data

async def send_array(writer: asyncio.StreamWriter, array: np.ndarray):
  # send number of dims, then size of each dim, then the array data
  writer.write(len(array.shape).to_bytes(4, byteorder="big"))
  for dim in array.shape:
    writer.write(dim.to_bytes(4, byteorder="big"))
  writer.write(array.tobytes())
  await writer.drain()  # Ensure data is sent

async def receive_array(reader: asyncio.StreamReader) -> np.ndarray:
  ndims = int.from_bytes(receive_exact(reader, 4), byteorder="big")
  shape = []
  for _ in range(ndims):
    dim_size = int.from_bytes(receive_exact(reader, 4), byteorder="big")
    shape.append(dim_size)
  assert len(shape) > 0
  array_size = functools.reduce(operator.mul, shape)
  array = receive_exact(reader, array_size)
  array = np.frombuffer(array, dtype=np.uint8).reshape(shape)
  return array