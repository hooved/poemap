from training_data import ViTDataLoader

dl = ViTDataLoader("data/train")

for i,layout in enumerate(dl.train_data):
  print(f"layout: {layout}")
  for instance in dl.train_data[layout]:
    print(f"instance: {instance}")
    count = len(dl.train_data[layout][instance])
    print(f"layout {layout}, instance {instance}: {count} paths")