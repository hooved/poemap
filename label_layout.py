import os, sys, shutil

if __name__=="__main__":
  layout_id = sys.argv[1]
  os.makedirs(layout_dir := f"data/train/{layout_id}", exist_ok=True)
  current_instances = tuple(int(x) for x in os.listdir(layout_dir))
  instance_id = 0 if not current_instances else max(current_instances) + 1
  shutil.move("data/train/collect", f"{layout_dir}/{instance_id}")
  