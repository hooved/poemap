import pyautogui
import keyboard
import time, os, sys
import numpy as np

def capture_screen():
  # player icon is at 1920, 1060, in 4k
  #box_radius = 300
  region = (1620, 760, 600, 600)
  screenshot = pyautogui.screenshot(region=region)
  return np.array(screenshot)
  #return screenshot.save("test.png")
  #img_byte_arr = io.BytesIO()
  #screenshot.save(img_byte_arr, format='JPEG', quality=50)  # Compress to JPEG
  #screenshot.save(img_byte_arr, format='PNG', compress_level=0)
  #return img_byte_arr.getvalue()

def stream_frames(target_fps=2):
  frame_time = 1 / target_fps
  frames = []
  print("Starting stream...")
  while True:
    last_capture_time = time.perf_counter()
    frame = capture_screen()
    frames.append(frame)
    print(f"{len(frames)} frames")

    if stop_stream or stop_program:
      print("Stopping stream...")
      print(f"number of frames: {len(frames)}")
      break

    elapsed = time.perf_counter() - last_capture_time
    if elapsed < frame_time:
      time.sleep(frame_time - elapsed)

def trigger_start_stream():
  global stop_stream
  stop_stream = False

def trigger_stop_stream():
  global stop_stream
  stop_stream = True

def trigger_stop_program():
  global stop_program
  stop_program = True

if __name__=="__main__":
  # using global vars because keyboard doesn't allow triggering one hotkey's callback during another
  stop_stream = True
  stop_program = False
  keyboard.add_hotkey("ctrl+alt+q", trigger_start_stream)
  keyboard.add_hotkey("ctrl+alt+w", trigger_stop_stream)
  keyboard.add_hotkey("ctrl+alt+e", trigger_stop_program)

  while True:
    if not stop_stream:
      stream_frames(target_fps=2)

    if stop_program:
      sys.exit()

    time.sleep(0.5)