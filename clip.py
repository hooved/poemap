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

from PIL import Image, ImageGrab
import numpy as np
import os

clip = ImageGrab.grabclipboard()
clip = np.array(clip)
white_pixels = (clip == [255, 255, 255]).all(axis=-1)
clip[white_pixels] = [0, 0, 0]
#Image.fromarray(clip).save("test.png")

os.makedirs("clips", exist_ok=True)
fp_list = os.listdir("clips")

if not fp_list:
  clip_id = 0
else:
  clip_id = max([int(os.path.splitext(fp)[0]) for fp in os.listdir("clips")]) + 1

np.savez_compressed(os.path.join("clips", f"{clip_id}.npz"), data=clip)