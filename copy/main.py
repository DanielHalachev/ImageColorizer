import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from models import MainModel
from dataset import make_dataloaders
from utils import create_loss_meters, update_losses, log_results, visualize

print(f"Libraries imported")

# Set up paths for data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = "/home/daniel/PycharmProjects/ImageColorizer/copy/data/tiny-imagenet-200/train"
image_pattern = os.path.join(path, '*', 'images', '*.JPEG')
print(image_pattern)
paths = glob.glob(image_pattern)
print(len(paths))

np.random.seed(123)
paths_subset = np.random.choice(paths, 10_000, replace=False)  # choosing 10 000 images randomly
rand_indexes = np.random.permutation(10_000)
train_indexes = rand_indexes[:8000]  # choosing the first 8000 as training set
validate_indexes = rand_indexes[8000:]  # choosing last 2000 as validation set
train_paths = paths_subset[train_indexes]
val_paths = paths_subset[validate_indexes]
print(f"Images withdrawn with their paths")

# Show some chosen images
_, axes = plt.subplots(4, 4, figsize=(10, 10))
for ax, img_path in zip(axes.flatten(), train_paths):
    ax.imshow(Image.open(img_path))
    ax.axis("off")

# Make dataloaders
train_dl = make_dataloaders(paths=train_paths, split='train')
val_dl = make_dataloaders(paths=val_paths, split='val')
print(f"Created dataloaders")


def train_model(model, train_dataloader, epochs, display_every=200):
    visualization_data = next(iter(val_dl))  # getting a batch for visualizing the model output after fixed intervals
    for e in range(epochs):
        loss_meter_dict = create_loss_meters()
        i = 0
        for data in tqdm(train_dataloader):
            model.setup_input(data)
            model.optimize()
            update_losses(model, loss_meter_dict, count=data['L'].size(0))
            i += 1
            if i % display_every == 0:
                print(f"\nEpoch {e + 1}/{epochs}")
                print(f"Iteration {i}/{len(train_dataloader)}")
                log_results(loss_meter_dict)  # function to print out the losses
                visualize(model, data, save=True)  # function displaying the model's outputs


main_model = MainModel()
print(f"Commence training")
train_model(main_model, train_dl, 100)
