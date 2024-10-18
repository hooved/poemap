import os, sys, shutil, glob

if __name__=="__main__":
  temp_dir = os.path.join("data", "train", "collect")
  layout_id = sys.argv[1]
  if layout_id == "c": # count how many instances have been collected per layout_id
    layouts = os.listdir(os.path.join("data", "train"))
    #layouts = dict(zip(layouts, [0]*len(layouts)))
    for layout in layouts:
      print(f"{layout}: {len(os.listdir(os.path.join('data', 'train', layout)))}")
  else:
    os.makedirs(layout_dir := os.path.join("data", "train", layout_id), exist_ok=True)
    current_instances = tuple(int(x) for x in os.listdir(layout_dir))
    instance_id = 0 if not current_instances else max(current_instances) + 1
    # Masks were inspected for quality, no longer needed for training
    #for file in glob.glob(os.path.join(temp_dir, "*_mask.png")):
      #os.remove(file)
    shutil.move(temp_dir, os.path.join(layout_dir, str(instance_id)))