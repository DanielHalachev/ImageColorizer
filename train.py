

def train_model(model, train_dataloader, epochs, display_every=100):
    # visualization_data = next(iter(val_dl))  # getting a batch for visualizing the model output after fixed intervals
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


if __name__ == "__main__":
    import glob
    import os
    import argparse

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm

    from models import MainModel
    from dataset import make_dataloaders
    from utils import create_loss_meters, update_losses, log_results, visualize

    EPOCHS = 100
    TRAINING_IMAGES = 15_000
    TOTAL_IMAGES = 16_000

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Start training a model')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset folder')
    args = parser.parse_args()

    # Detect and set up device to compute
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using a {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"Using {torch.cuda.device_count()} devices")

    # Set up paths for data
    path = os.path.abspath(args.dataset_path)
    image_pattern = os.path.join(path, '*')
    print(f"Getting images by regex {image_pattern}")
    paths = glob.glob(image_pattern)
    print(f"A total of {len(paths)} paths were extracted")

    # Forming datasets
    np.random.seed(500)
    paths_subset = np.random.choice(paths, TOTAL_IMAGES, replace=False)
    rand_indexes = np.random.permutation(TOTAL_IMAGES)
    train_indexes = rand_indexes[:TRAINING_IMAGES]
    validate_indexes = rand_indexes[TRAINING_IMAGES:]
    train_paths = paths_subset[train_indexes]
    val_paths = paths_subset[validate_indexes]
    print(f"Images withdrawn with their paths")

    # # Show some chosen images
    # _, axes = plt.subplots(4, 4, figsize=(10, 10))
    # for ax, img_path in zip(axes.flatten(), train_paths):
    #     ax.imshow(Image.open(img_path))
    #     ax.axis("off")

    # Make dataloaders
    train_dl = make_dataloaders(paths=train_paths, split='train')
    val_dl = make_dataloaders(paths=val_paths, split='val')
    print(f"Created dataloaders")

    main_model = MainModel()
    print(f"Training commenced")
    train_model(main_model, train_dl, EPOCHS)

    # saving weights for later
    torch.save(main_model.state_dict(), "model.pt")
