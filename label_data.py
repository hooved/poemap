import os, sys, shutil, glob, json
from collections import defaultdict
from typing import List

if __name__=="__main__":
  temp_dir = os.path.join("data", "train", "collect")
  layout_id = sys.argv[1]

  if layout_id == "c": # count how many instances have been collected per layout_id
    layouts = os.listdir(os.path.join("data", "test"))
    #layouts = dict(zip(layouts, [0]*len(layouts)))
    for layout in layouts:
      if layout != "curation_mask.json":
        print(f"{layout}: {len(os.listdir(os.path.join('data', 'test', layout)))}")

  else:

    os.makedirs(layout_dir := os.path.join("data", "test", layout_id), exist_ok=True)
    current_instances = tuple(int(x) for x in os.listdir(layout_dir))
    #layout_id = int(layout_id)
    # json requires str keys
    instance_id = "0" if not current_instances else str(max(current_instances) + 1)
    os.mkdir(dest_dir := os.path.join(layout_dir, instance_id))
    for file in glob.glob(os.path.join(temp_dir, "*")):
      shutil.move(file, dest_dir)
    shutil.rmtree(temp_dir)

    # manually curated high quality samples for eval
    selections = sys.argv[2:]


    # ugly json load/save code; this just collates our above selections with all previous selections
    if os.path.exists(mask_fp := os.path.join("data", "test", "curation_mask.json")):
      with open(mask_fp, "r") as json_file:
        mask = json.load(json_file)

        # convert Dict[Dict[list]] to DefaultDict[DefaultDict[List]], json is incompatible with the latter
        for l_id, instances in mask.items():
          mask[l_id] = defaultdict(list, instances)
        mask = defaultdict(lambda: defaultdict(list), mask)

    else:
      mask = defaultdict(lambda: defaultdict(list))

    mask[layout_id][instance_id] = selections # List[str] of integer timestamps

    # convert back to Dict[Dict[list]]
    for l_id, instances in mask.items():
      mask[l_id] = dict(instances)
    mask = dict(mask)

    with open(mask_fp, "w") as json_file:
      json.dump(mask, json_file)
