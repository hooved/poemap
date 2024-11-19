from training_data import ViTDataLoader

dl = ViTDataLoader("data/train")

for i,layout in enumerate(dl.train_data):
  layout_sum = 0
  for instance in dl.train_data[layout]:
    count = len(dl.train_data[layout][instance])
    print(f"layout {layout}, instance {instance}: {count} paths")
    layout_sum += count
  print(f"layout {layout}, total: {layout_sum} paths")

X_steps, Y_steps = dl.get_epoch(min_samples_per_class_per_step=5)
X_test, Y_test = dl.get_test_data()

done = 1