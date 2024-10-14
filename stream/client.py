import pyautogui
import keyboard
import time, os, sys, socket, asyncio
from comms import send_array
import numpy as np
import cv2
from PIL import Image

class AsyncState:
  def __init__(self):
    self.data = None
    self.lock = asyncio.Lock()
    self.data_ready = asyncio.Event()
    self.stop_stream = True
    self.stop_program = False
    keyboard.add_hotkey("ctrl+alt+q", self.trigger_start_stream)
    keyboard.add_hotkey("ctrl+alt+w", self.trigger_stop_stream)
    keyboard.add_hotkey("ctrl+alt+e", self.trigger_stop_program)

  async def write(self, data):
    async with self.lock:
      self.data = data
      self.data_ready.set()

  async def read(self):
    await self.data_ready.wait()
    async with self.lock:
      self.data_ready.clear()
      return self.data

  def trigger_start_stream(self):
    self.stop_stream = False

  def trigger_stop_stream(self):
    self.stop_stream = True

  def trigger_stop_program(self):
    self.stop_program = True

def capture_screen(box_radius):
  # player icon is at 1920, 1060, in 4k
  region = (1920 - box_radius, 1060 - box_radius, box_radius*2, box_radius*2)
  screenshot = pyautogui.screenshot(region=region)
  return np.array(screenshot)

async def produce_minimap(state: AsyncState, target_fps=2, box_radius=600):
  frame_time = 1 / target_fps
  minimap = capture_screen(box_radius)
  frames = [minimap]
  origin = (box_radius, box_radius)
  last_position = origin
  print("Starting stream...")
  while True:
    last_capture_time = time.perf_counter()
    frame = capture_screen(box_radius)
    frames.append(frame)
    minimap, origin, last_position = minimap_append_frame(minimap, np.array(frames), origin, last_position)
    frames.pop(0)
    await state.write(minimap)

    if state.stop_stream or state.stop_program:
      print("Stopping stream...")
      return

    elapsed = time.perf_counter() - last_capture_time
    if elapsed < frame_time:
      await asyncio.sleep(frame_time - elapsed)

async def consume_minimap(state: AsyncState):
  while True:
    minimap = await state.read()
    cv2.namedWindow('Overlay', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Overlay', cv2.WND_PROP_TOPMOST, 1)
    #cv2.moveWindow('Overlay', 3840-minimap.shape[1], 700)
    cv2.imshow('Overlay', minimap)
    cv2.waitKey(1)
    if state.stop_stream or state.stop_program:
      cv2.destroyAllWindows()
      return
    await asyncio.sleep(3)

# From https://github.com/kweimann/poe-learning-layouts/blob/main/utils/data.py, MIT License
def find_translation(image_a, image_b, threshold=0.7, max_matches=15):
  # determine the (dx, dy) vector required to move from image_a to image_b
  # noinspection PyUnresolvedReferences
  sift = cv2.SIFT_create()
  # convert images to grayscale
  image_a_gray = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
  image_b_gray = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
  # find SIFT features
  keypoints_a, descriptors_a = sift.detectAndCompute(image_a_gray, None)
  keypoints_b, descriptors_b = sift.detectAndCompute(image_b_gray, None)
  if len(keypoints_a) == 0 or len(keypoints_b) == 0:
    return None
  # match the SIFT features
  matcher = cv2.BFMatcher(crossCheck=False)
  matches = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
  # filter bad matches and sort
  matches = [
    match[0] for match in matches
    if len(match) == 2 and match[0].distance < threshold * match[1].distance
  ]
  if len(matches) == 0:
    return None  # failed to find translation
  matches = sorted(matches, key=lambda m: m.distance)[:max_matches]
  # select the matching points
  points_a = np.array([keypoints_a[m.queryIdx].pt for m in matches]).reshape(-1, 2)
  points_b = np.array([keypoints_b[m.trainIdx].pt for m in matches]).reshape(-1, 2)
  # get the mean translation
  dx, dy = np.median(points_a - points_b, axis=0)
  return dx, dy

def minimap_append_frame(minimap, diff_frames, origin, last_position):
  assert len(diff_frames) == 2
  _, H_frame, W_frame, C = diff_frames.shape  # frame size
  assert H_frame % 2 == 0 and W_frame % 2 == 0
  old_H, old_W, _ = minimap.shape

  # dx, dy
  move = np.round(find_translation(diff_frames[0], diff_frames[1])).astype(int)
  # y, x
  current_position = (last_position[0] + move[1], last_position[1] + move[0])

  pad_up = abs(min(current_position[0] - H_frame//2, 0))
  pad_down = abs(min(old_H - current_position[0] - H_frame//2, 0))
  pad_left = abs(min(current_position[1] - W_frame//2, 0))
  pad_right = abs(min(old_W - current_position[1] - W_frame//2, 0))
  if not any((pad_up, pad_down, pad_left, pad_right)):
    return minimap, origin, current_position

  minimap = np.pad(minimap, ((pad_up, pad_down), (pad_left, pad_right), (0,0)), mode='constant')
  origin = (origin[0] + pad_up, origin[1] + pad_left)
  current_position = (current_position[0] + pad_up, current_position[1] + pad_left)
  frame_top = current_position[0] - H_frame//2
  frame_left = current_position[1] - W_frame//2
  assert frame_top >= 0 and frame_left >=0
  minimap[frame_top : frame_top+H_frame, frame_left : frame_left+W_frame] = diff_frames[1]
  return minimap, origin, current_position

async def main():
  state = AsyncState()

  while True:
    if not state.stop_stream:
      async with asyncio.TaskGroup() as tg:
        producer = tg.create_task(produce_minimap(state, target_fps=1))
        consumer = tg.create_task(consume_minimap(state))

    if state.stop_program:
      sys.exit()

    time.sleep(0.5)

if __name__=="__main__":
  #client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  #client.connect(('localhost', 50000))
  #array = np.array(Image.open("minimap.png"))
  #send_array(client, array)
  #client.close()
  #sys.exit()

  asyncio.run(main())