import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

SIZE = 256


class ColorizationDataset(Dataset):
    """
    Custom class for loading the dataset into the neural network using PyTorch
    """

    def __init__(self, paths, split='train'):
        """
        Colorization Dataset initializer
        :param paths: List of file paths for images
        :type paths: list
        :param split:Dataset split, either 'train' or 'test'
        :type split: str, optional
        """
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((SIZE, SIZE), Image.BICUBIC),
                transforms.RandomHorizontalFlip(),  # A little data augmentation!
            ])
        elif split == 'val':
            self.transforms = transforms.Resize((SIZE, SIZE), Image.BICUBIC)

        self.split = split
        self.size = SIZE
        self.paths = paths

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")  # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1.  # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110.  # Between -1 and 1

        return {'L': L, 'ab': ab}

    def __len__(self):
        return len(self.paths)


def make_dataloaders(batch_size=16, n_workers=4, pin_memory=True, **kwargs):
    """
        Create DataLoader for ColorizationDataset.

        :param batch_size: Number of samples per batch. Default is 16.
        :type batch_size: int, optional
        :param n_workers: Number of workers for data loading. Default is 4.
        :type n_workers: int, optional
        :param pin_memory: Whether to use pinned memory for faster data transfer. Default is True.
        :type pin_memory: bool, optional
        :param kwargs: Additional arguments to pass to ColorizationDataset.

        :return: DataLoader for the ColorizationDataset.
        :rtype: torch.utils.data.DataLoader
    """
    dataset = ColorizationDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, pin_memory=pin_memory)
    return dataloader
