from PIL import Image
import numpy as np
from stream.client import minimap_append_frame
from training_data import crop_frame

if __name__=="__main__":
  frame_1 = crop_frame(np.array(Image.open("data/train/3/2/0.png")), 600)
  frame_2 = crop_frame(np.array(Image.open("data/train/3/2/1.png")), 600)
  minimap = frame_1
  origin = (600, 600)
  last_position = origin
  diff_frames = np.array([frame_1, frame_2])

  minimap, origin, last_position = minimap_append_frame(minimap, diff_frames, origin, last_position)
  Image.fromarray(minimap).save("data/train/3/2/0_1.png")
  done = 1