import pyautogui
import io
import socket
import time

def capture_screen():
  # player icon is at 1920, 1060, in 4k
  #box_radius = 300
  region = (1620, 760, 600, 600)
  screenshot = pyautogui.screenshot(region=region)
  return screenshot.save("test.png")
  img_byte_arr = io.BytesIO()
  #screenshot.save(img_byte_arr, format='JPEG', quality=50)  # Compress to JPEG
  screenshot.save(img_byte_arr, format='PNG', compress_level=0)
  return img_byte_arr.getvalue()

HOST = 'localhost'
#PORT = 65432  # Port to listen on
#PORT = 1023  # Port to listen on
PORT = 50000  # Port to listen on

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
  s.bind((HOST, PORT))
  s.listen(1)
  print('Waiting for connection...')
  conn, addr = s.accept()
  with conn:
    print('Connected by', addr)
    assert 1==0
    while True:
      frame_data = capture_screen()
      data_length = len(frame_data).to_bytes(4, byteorder='big')
      conn.sendall(data_length + frame_data)
      time.sleep(0.1)  # Adjust frame rate as needed
