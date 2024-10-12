from PIL import Image
import numpy as np
from stream.client import minimap_append_frame
from training_data import crop_frame

if __name__=="__main__":
  frame_1 = crop_frame(np.array(Image.open("data/train/3/2/0.png")), 600)
  minimap = frame_1
  origin = (600, 600)
  last_position = origin

  for i in range(1, 4):
    frame_1 = crop_frame(np.array(Image.open(f"data/train/3/2/{i-1}.png")), 600)
    frame_2 = crop_frame(np.array(Image.open(f"data/train/3/2/{i}.png")), 600)
    diff_frames = np.array([frame_1, frame_2])
    minimap, origin, last_position = minimap_append_frame(minimap, diff_frames, origin, last_position)
    Image.fromarray(minimap).save(f"data/train/3/2/0_{i}.png")
  done = 1