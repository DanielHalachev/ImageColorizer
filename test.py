import argparse
import torch
import torchvision.transforms as transforms
import PIL.Image
import matplotlib.pyplot as plt

from models import MainModel
from utils import lab_to_rgb


def test_image_with_model(model_path, image_path):
    """
    The function reconstructs a model and runs an image through it. The colorized result is displayed.
    :param model_path: The path to the model weights
    :type model_path: str
    :param image_path: The path to the grayscale image to be colorized
    :type image_path: str
    :return: None
    """
    model = MainModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    img = PIL.Image.open(image_path)
    img = img.resize((256, 256))
    img = transforms.ToTensor()(img)[:1] * 2. - 1.
    model.eval()
    with torch.no_grad():
        predictions = model.net_G(img.unsqueeze(0).to(device))
    colorized = lab_to_rgb(img.unsqueeze(0), predictions.cpu())[0]
    plt.imshow(colorized)
    plt.show()


if __name__ == '__main__':
    """
    This script can be used to colorize a grayscale image using a pretrained model. The script takes two arguments - 
    the path to the pretrained model and the path to the image to be colorized
    """
    parser = argparse.ArgumentParser(description='Test an image with a trained model')
    parser.add_argument('model_path', type=str, help='Path to the model weights file')
    parser.add_argument('image_path', type=str, help='Path to the input image file')
    args = parser.parse_args()

    test_image_with_model(args.model_path, args.image_path)
