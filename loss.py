import torch
from torch import nn


class GANLoss(nn.Module):
    """
    A module for measuring the GAN Loss.
    """
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        """
        Initializes the GANLoss module.

        :param real_label: The label value for real samples (default is 1.0).
        :type real_label: float
        :param fake_label: The label value for fake/generated samples (default is 0.0).
        :type fake_label: float
        """
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()

    def get_labels(self, preds, target_is_real):
        """
        Returns the target labels based on whether the target is real or fake.

        :param preds:The predictions from the discriminator.
        :type preds: torch.Tensor
        :param target_is_real:Indicates whether the target is a real sample.
        :type target_is_real: bool

        :return: The target labels expanded to match the shape of predictions.
        :rtype: torch.Tensor
        """
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)

    def __call__(self, preds, target_is_real):
        """
        Computes and returns the adversarial loss given the predictions and target labels.

        :param preds: The predictions from the discriminator.
        :type preds: torch.Tensor
        :param target_is_real: Indicates whether the target is a real sample.
        :type target_is_real: bool

        :return: The computed adversarial loss.
        :rtype: torch.Tensor
        """
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss
