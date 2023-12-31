import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.color import lab2rgb


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count


def create_loss_meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()

    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G}


def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)


def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """

    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


def visualize(model, data, save=True):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        # grayscale
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        #
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")

        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    plt.show(block=False)
    if save:
        base_directory = os.getcwd()
        save_folder = os.path.join(base_directory, 'data', 'stages')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        file_name = f"colorization_{time.time()}.png"
        file_path = os.path.join(save_folder, file_name)
        fig.savefig(file_path)


def log_results(loss_meter_dict):
    base_directory = os.getcwd()
    data_folder = os.path.join(base_directory, 'data')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    csv_file = os.path.join(data_folder, 'losses.csv')

    file_exists = os.path.isfile(csv_file)
    headers = list(loss_meter_dict.keys())  # Using the loss names as headers
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)

        if not file_exists:
            writer.writeheader()
        log_data = {loss_name: f"{loss_meter.avg:.5f}" for loss_name, loss_meter in loss_meter_dict.items()}
        writer.writerow(log_data)
