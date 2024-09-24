import socket
import numpy as np
import functools, operator

class Socket(socket.socket):
  def receive_exact(self, n):
    data = b''
    while len(data) < n:
      packet = self.recv(n - len(data))
      if not packet:
        return None
      data += packet
    return data

  def send_array(self, array: np.ndarray):
    # send number of dims, then size of each dim, then the array data
    self.send(len(array.shape.to_bytes(4, byteorder="big")))
    for dim in array.shape:
      self.send(dim.to_bytes(4, byteorder="big"))
    self.sendall(array.to_bytes())

  def receive_array(self):
    ndims = int.from_bytes(self.receive_exact(4), byteorder="big")
    shape = []
    for _ in range(ndims):
      dim_size = int.from_bytes(self.receive_exact(4), byteorder="big")
      shape.append(dim_size)
    assert len(shape) > 0
    array_size = functools.reduce(operator.mul, shape)
    array = self.receive_exact(array_size)
    array = np.frombuffer(array, dtype=np.uint8).reshape(shape)
    return array