# This script saves the line drawn on an MSPaint canvas
# Each line represents a path through a layout
# Later we will automatically sample only the parts of the image 
  # near the line (representing light radius around player's path in game),
  # and we will use these samplings to train the layout classifier,
  # representing realistic samplings a player would take at test time

# First make "draw" dir in poemap folder, 
## for each layout to be sampled:
  # copy PNG for layout to be sampled to poemap/draw/layout.png,
  ## for each path you want to sample:
    # draw a pure yellow line (rgb = (0, 255, 255)) on the PNG in paint
    # use the clip.ahk script to call this script
    # (this script automates the saving of the line coordinates to poemap/draw/paths.npz)
  # copy poemap/draw/paths.npz back to the original layout dir
  # rm poemap/draw/* (to prepare for next layout to sample)

from PIL import ImageGrab
import numpy as np
import os

clip = ImageGrab.grabclipboard()
clip = np.array(clip)
yellow_pixels = (clip == [255, 255, 0]).all(axis=-1)
yellow_pixels = np.argwhere(yellow_pixels)

paths_fp = os.path.join("draw", "paths.npz")
if os.path.exists(paths_fp):
  paths = list(np.load(paths_fp, allow_pickle=True)['paths'])
else:
  paths = []

paths.append(yellow_pixels)
paths = np.array(paths, dtype=object)

assert yellow_pixels.shape[0] > 0
print(f"saving yellow_pixels with shape: {yellow_pixels.shape}")
np.savez_compressed(paths_fp, paths=paths)