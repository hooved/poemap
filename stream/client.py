#import pyautogui
#import keyboard
import time, os, sys, socket
from .comms import send_array
import numpy as np
import cv2
from PIL import Image

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

    if len(frames) == 32:
      stop_stream = True

    if len(frames) % 8 == 0:
      minimap = frames_to_map(np.array(frames))
      #mask = map_to_mask(minimap)
      cv2.namedWindow('Overlay', cv2.WINDOW_NORMAL)
      cv2.setWindowProperty('Overlay', cv2.WND_PROP_TOPMOST, 1)
      #cv2.moveWindow('Overlay', 3840-minimap.shape[1], 700)
      cv2.imshow('Overlay', minimap)
      cv2.waitKey(1)

    if stop_stream or stop_program:
      print("Stopping stream...")
      cv2.destroyAllWindows()
      #print(f"number of frames: {len(frames)}")
      #minimap = frames_to_map(np.array(frames))
      #Image.fromarray(minimap).save("minimap.png")
      return

    elapsed = time.perf_counter() - last_capture_time
    if elapsed < frame_time:
      time.sleep(frame_time - elapsed)

def frames_to_map(frames):
  moves = get_moves(frames)
  return draw_minimap(frames, moves)

def get_moves(frames):
  moves = [(0,0)]
  for i, frame in enumerate(frames[1:]):
      move = find_translation(frames[i], frame)
      moves.append(move)
  moves = np.round(np.array(moves)).astype(int)
  return moves

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

# From https://github.com/kweimann/poe-learning-layouts/blob/main/utils/data.py, MIT License
def draw_minimap(video, movement):
  current_position = movement.cumsum(axis=0)
  x_min, y_min = current_position.min(axis=0)
  x_max, y_max = current_position.max(axis=0)
  assert (x_min <= 0) and (x_max >= 0)
  assert (y_min <= 0) and (y_max >= 0)
  (x_start, y_start) = abs(x_min), abs(y_min)
  _, H_frame, W_frame, C = video.shape  # frame size
  (H, W) = (y_max - y_min + H_frame, x_max - x_min + W_frame)  # map size
  H += H % 2  # make height divisible by 2
  W += W % 2  # make width divisible by 2
  minimap = np.zeros((H, W, C), dtype=np.uint8)
  for frame, (x, y) in zip(video, current_position):
    minimap[y_start + y:y_start + y + H_frame, x_start + x:x_start + x + W_frame] = frame
  origin = (y_start + H_frame // 2, x_start + W_frame // 2)
  return minimap, origin

def minimap_append_frame(minimap, diff_frames, origin, last_position):
  assert len(diff_frames) == 2
  _, H_frame, W_frame, C = diff_frames.shape  # frame size
  assert H_frame % 2 == 0 and W_frame % 2 == 0
  old_H, old_W, _ = minimap.shape

  move = np.round(find_translation(diff_frames[0], diff_frames[1])).astype(int)
  current_position = (last_position[0] + move[0], last_position[1] + move[1])

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
  return minimap, origin, last_position

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
  client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  client.connect(('localhost', 50000))
  array = np.array(Image.open("minimap.png"))
  send_array(client, array)
  client.close()
  sys.exit()

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