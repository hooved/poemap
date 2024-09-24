import numpy as np
import functools, operator

def receive_exact(sock, n):
  data = b''
  while len(data) < n:
    packet = sock.recv(n - len(data))
    if not packet:
      return None
    data += packet
  return data

def send_array(sock, array: np.ndarray):
  # send number of dims, then size of each dim, then the array data
  sock.send(len(array.shape).to_bytes(4, byteorder="big"))
  for dim in array.shape:
    sock.send(dim.to_bytes(4, byteorder="big"))
  sock.sendall(array.tobytes())

def receive_array(sock):
  ndims = int.from_bytes(receive_exact(sock, 4), byteorder="big")
  shape = []
  for _ in range(ndims):
    dim_size = int.from_bytes(receive_exact(sock, 4), byteorder="big")
    shape.append(dim_size)
  assert len(shape) > 0
  array_size = functools.reduce(operator.mul, shape)
  array = receive_exact(sock, array_size)
  array = np.frombuffer(array, dtype=np.uint8).reshape(shape)
  return array