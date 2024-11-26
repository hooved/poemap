import pyautogui
import keyboard
import time, os, sys, asyncio, glob
from comms import receive_int, send_array, receive_array, ClientHeader
from helpers import shrink_with_origin
import numpy as np
import cv2
from PIL import Image
from typing import Dict

class AsyncState:
  def __init__(self):
    self.data = None
    self.lock = asyncio.Lock()
    self.data_ready = asyncio.Event()
    self.stream_minimap = False
    self.stream_frames = False
    self.stop_stream = True
    self.stop_program = False
    keyboard.add_hotkey("ctrl+alt+q", self.trigger_stream_minimap)
    keyboard.add_hotkey("ctrl+alt+z", self.trigger_stream_frames)
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

  def trigger_stream_minimap(self):
    self.stop_stream = False
    self.stream_minimap = True

  def trigger_stream_frames(self):
    self.stop_stream = False
    self.stream_frames = True

  def trigger_stop_stream(self):
    self.stop_stream = True
    self.stream_minimap = False
    self.stream_frames = False

  def trigger_stop_program(self):
    self.stop_program = True

def capture_screen(box_radius):
  # player icon is at 1920, 1060, in 4k
  region = (1920 - box_radius, 1060 - box_radius, box_radius*2, box_radius*2)
  screenshot = pyautogui.screenshot(region=region)
  return np.array(screenshot)

async def produce_minimap(state: AsyncState, target_fps=0.5, box_radius=600):
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
    await state.write(shrink_with_origin(minimap, origin))

    if state.stop_stream or state.stop_program:
      print("Stopping stream...")
      return

    elapsed = time.perf_counter() - last_capture_time
    if elapsed < frame_time:
      await asyncio.sleep(frame_time - elapsed)

async def consume_minimap(state: AsyncState):
  reader, writer = await asyncio.open_connection('localhost', 50000)
  while True:
    minimap, origin = await state.read()
    await send_header(writer, ClientHeader.PROCESS_MINIMAP)
    await send_array(writer, minimap)
    await send_array(writer, np.array(origin, dtype=np.uint32))
    try:
      layout_id = await receive_int(reader)
    except:
      Image.fromarray(minimap).save("failed_minimap.png")
      assert False
    print(f"layout_id = {layout_id}")
    display_layout(layout_guides[layout_id])

    if state.stop_stream or state.stop_program:
      cv2.destroyAllWindows()
      await send_header(writer, ClientHeader.CLOSE_CONNECTION)
      writer.close()
      await writer.wait_closed()
      return

def display_layout(layout: np.ndarray, max_width=1200, max_height=1000):
    height, width = layout.shape[:2]
    scale = min(max_width / width, max_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    layout_resized = cv2.resize(layout, (new_width, new_height), interpolation=cv2.INTER_AREA)

    cv2.namedWindow('Overlay', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Overlay', new_width, new_height)
    cv2.setWindowProperty('Overlay', cv2.WND_PROP_TOPMOST, 1)
    cv2.imshow('Overlay', cv2.cvtColor(layout_resized, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)

async def send_header(writer: asyncio.StreamWriter, header: ClientHeader):
  writer.write(header.to_bytes(4, byteorder="big"))
  await writer.drain()

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
  #if not any((pad_up, pad_down, pad_left, pad_right)):
    #return minimap, origin, current_position
  minimap = np.pad(minimap, ((pad_up, pad_down), (pad_left, pad_right), (0,0)), mode='constant')
  origin = (origin[0] + pad_up, origin[1] + pad_left)
  current_position = (current_position[0] + pad_up, current_position[1] + pad_left)
  frame_top = current_position[0] - H_frame//2
  frame_left = current_position[1] - W_frame//2
  assert frame_top >= 0 and frame_left >=0
  minimap[frame_top : frame_top+H_frame, frame_left : frame_left+W_frame] = diff_frames[1]
  return minimap, origin, current_position

async def send_frames(state: AsyncState, target_fps=0.5, box_radius=600):
  reader, writer = await asyncio.open_connection('localhost', 50000)
  frame_time = 1 / target_fps
  print("Starting stream...")
  while True:
    last_capture_time = time.perf_counter()
    frame = capture_screen(box_radius)
    await send_header(writer, ClientHeader.SAVE_FRAME)
    await send_array(writer, frame)
    finished = await receive_int(reader)
    assert finished == 42

    if state.stop_stream or state.stop_program:
      print("Stopping stream...")
      await send_header(writer, ClientHeader.CLOSE_CONNECTION)
      writer.close()
      await writer.wait_closed()
      return

    elapsed = time.perf_counter() - last_capture_time
    if elapsed < frame_time:
      await asyncio.sleep(frame_time - elapsed)

async def main():
  state = AsyncState()

  while True:
    if not state.stop_stream:

      if state.stream_minimap:
        async with asyncio.TaskGroup() as tg:
          producer = tg.create_task(produce_minimap(state, target_fps=1))
          consumer = tg.create_task(consume_minimap(state))
          state = AsyncState()

      elif state.stream_frames:
        async with asyncio.TaskGroup() as tg:
          producer = tg.create_task(send_frames(state, target_fps=1))
          state = AsyncState()

    if state.stop_program:
      sys.exit()

    await asyncio.sleep(0.1)

if __name__=="__main__":
  layout_guides = {}
  for layout_fp in glob.glob(os.path.join("data", "user", "coast", "*.png")):
    layout_id = int(os.path.relpath(layout_fp, os.path.dirname(layout_fp)).split(".png")[0])
    layout_guides[layout_id] = np.array(Image.open(layout_fp))

  asyncio.run(main())