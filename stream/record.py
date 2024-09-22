import pyautogui
import keyboard
import time
import numpy as np

def capture_screen():
  # player icon is at 1920, 1060, in 4k
  #box_radius = 300
  region = (1620, 760, 600, 600)
  screenshot = pyautogui.screenshot(region=region)
  #return screenshot.save("test.png")
  #img_byte_arr = io.BytesIO()
  #screenshot.save(img_byte_arr, format='JPEG', quality=50)  # Compress to JPEG
  #screenshot.save(img_byte_arr, format='PNG', compress_level=0)
  #return img_byte_arr.getvalue()

def stream_frames():
  frames = []
  while True:
    frame = pyautogui.screenshot()
    frame = np.array(frame)
    frames.append(frame)

    if keyboard.is_pressed('ctrl+alt+w'):
      print("Stopping stream...")
      print(f"number of frames: {len(frames)}")
      break

if __name__=="__main__":
  while True:
    if keyboard.is_pressed('ctrl+alt+q'):
      print("Starting stream...")
      stream_frames()
    
    if keyboard.is_pressed('esc'):
      print("Exiting program...")
      break

    time.sleep(0.1)