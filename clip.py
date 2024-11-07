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