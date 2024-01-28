import torch
from torch import nn, optim
from loss import GANLoss


class UnetBlock(nn.Module):
    """
        Defines a U-Net block, a fundamental building block for the generator in an image-to-image translation model.
    """
    def __init__(self, nf, ni, submodule=None, input_channels=None, dropout=False, innermost=False, outermost=False):
        """
        Initialize UnetBlock.

        :param nf: Number of filters.
        :type nf: int
        :param ni: Number of input channels.
        :type ni: int
        :param submodule: Submodule to be included inside the block. Default is None.
        :type submodule: nn.Module, optional
        :param input_channels: Number of input channels. Default is None.
        :type input_channels: int, optional
        :param dropout: Whether to apply dropout. Default is False.
        :type dropout: bool, optional
        :param innermost: Whether the block is innermost. Default is False.
        :type innermost: bool, optional
        :param outermost: Whether the block is outermost. Default is False.
        :type outermost: bool, optional
        """
        super().__init__()
        self.outermost = outermost
        if input_channels is None:
            input_channels = nf
        down_convolution = nn.Conv2d(input_channels, ni, kernel_size=4, stride=2, padding=1, bias=False)
        down_relu = nn.LeakyReLU(0.2, True)
        down_normalization = nn.BatchNorm2d(ni)
        up_relu = nn.ReLU(True)
        up_norm = nn.BatchNorm2d(nf)

        if outermost:
            up_convolution = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4, stride=2, padding=1)
            down = [down_convolution]
            up = [up_relu, up_convolution, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            up_convolution = nn.ConvTranspose2d(ni, nf, kernel_size=4, stride=2, padding=1, bias=False)
            down = [down_relu, down_convolution]
            up = [up_relu, up_convolution, up_norm]
            model = down + up
        else:
            up_convolution = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4, stride=2, padding=1, bias=False)
            down = [down_relu, down_convolution, down_normalization]
            up = [up_relu, up_convolution, up_norm]
            if dropout:
                up += [nn.Dropout(0.5)]
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        """
        Forward pass through the U-Net block.

        :param x: Input tensor.
        :type x: torch.Tensor

        :return: Output tensor.
        :rtype: torch.Tensor
        """
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class Unet(nn.Module):
    """
    Defines a U-Net model constructed using U-Net Blocks
    """
    def __init__(self, nfg=64):
        super().__init__()
        unet_block = UnetBlock(nfg * 8, nfg * 8, innermost=True)
        for _ in range(3):
            unet_block = UnetBlock(nfg * 8, nfg * 8, submodule=unet_block, dropout=True)
        out_filters = nfg * 8
        for _ in range(3):
            unet_block = UnetBlock(out_filters // 2, out_filters, submodule=unet_block)
            out_filters //= 2
        self.model = UnetBlock(2, out_filters, input_channels=1, submodule=unet_block, outermost=True)

    def forward(self, x):
        """
        Forward pass through the U-Net model.

        :param x: Input tensor.
        :type x: torch.Tensor

        :return: Output tensor.
        :rtype: torch.Tensor
        """
        return self.model(x)


class PatchDiscriminator(nn.Module):
    """
    Defines a PatchGAN discriminator for image-to-image translation tasks.
    """
    def __init__(self, nfd=64):
        """
        Initialize PatchDiscriminator.

        :param nfd: Number of initial filters. Default is 64.
        :type nfd: int, optional
        """
        super().__init__()
        # No normalization in first block
        model = [self.get_layers(3, nfd, normalization=False)]
        # no stride = 2 in last block
        model += [self.get_layers(nfd * 2 ** i, nfd * 2 ** (i + 1), s=1 if i == 2 else 2)
                  for i in range(3)]
        # No normalization or action layer in last block
        model += [self.get_layers(nfd * 2 ** 3, 1, s=1, normalization=False, action=False)]
        self.model = nn.Sequential(*model)

    def get_layers(self, ni, nf, k=4, s=2, p=1, normalization=True, action=True):
        """
        Helper method to create layers for the PatchGAN discriminator.

        :param ni: Number of input channels.
        :type ni: int
        :param nf: Number of filters.
        :type nf: int
        :param k: Kernel size. Default is 4.
        :type k: int, optional
        :param s: Stride. Default is 2.
        :type s: int, optional
        :param p: Padding. Default is 1.
        :type p: int, optional
        :param normalization: Whether to apply batch normalization. Default is True.
        :type normalization: bool, optional
        :param action: Whether to apply activation function. Default is True.
        :type action: bool, optional

        :return: Sequential model representing the layers.
        :rtype: nn.Sequential
        """
        layers = [
            nn.Conv2d(ni, nf, k, s, p, bias=not normalization)]
        if normalization:
            layers += [nn.BatchNorm2d(nf)]
        if action:
            layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the PatchGAN discriminator.

        :param x: Input tensor.
        :type x: torch.Tensor

        :return: Output tensor.
        :rtype: torch.Tensor
        """
        return self.model(x)


def init_weights(net, init='norm', gain=0.02):
    """
    Initialize weights for the neural network.

    :param net: Neural network model.
    :type net: nn.Module
    :param init: Initialization method. Default is 'norm'.
    :type init: str, optional
    :param gain: Gain factor for weight initialization. Default is 0.02.
    :type gain: float, optional

    :return: Initialized neural network model.
    :rtype: nn.Module
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)

    net.apply(init_func)
    print(f"model initialized with {init} initialization")
    return net


def init_model(model, device):
    """
    Initialize a neural network model.

    :param model: Neural network model.
    :type model: nn.Module
    :param device: Device to which the model will be moved.
    :type device: torch.device

    :return: Initialized and moved neural network model.
    :rtype: nn.Module
    """
    model = model.to(device)
    model = init_weights(model)
    return model


class MainModel(nn.Module):
    """
    Main model for image-to-image translation tasks using a conditional GAN with a U-Net generator.
    """
    def __init__(self,
                 net_generator=None,
                 lr_generator=2e-4,
                 lr_discriminator=2e-4,
                 beta1=0.5,
                 beta2=0.999,
                 lambda_l1=100.):
        """
        Initialize MainModel.

        :param net_generator: Predefined generator network. Default is None.
        :type net_generator: nn.Module, optional
        :param lr_generator: Learning rate for the generator. Default is 2e-4.
        :type lr_generator: float, optional
        :param lr_discriminator: Learning rate for the discriminator. Default is 2e-4.
        :type lr_discriminator: float, optional
        :param beta1: Beta1 parameter for Adam optimizer. Default is 0.5.
        :type beta1: float, optional
        :param beta2: Beta2 parameter for Adam optimizer. Default is 0.999.
        :type beta2: float, optional
        :param lambda_l1: Weight for L1 loss term. Default is 100.
        :type lambda_l1: float, optional
        """
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_l1

        if net_generator is None:
            self.net_G = init_model(Unet(nfg=64), self.device)
        else:
            self.net_G = net_generator.to(self.device)
        self.net_D = init_model(PatchDiscriminator(nfd=64), self.device)
        self.GAN_criterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1_criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_generator, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_discriminator, betas=(beta1, beta2))

    def set_requires_grad(self, model, requires_grad=True):
        """
        Set the requires_grad attribute for model parameters.

        :param model: Model for which to set the requires_grad attribute.
        :type model: nn.Module
        :param requires_grad: Whether to set requires_grad to True or False. Default is True.
        :type requires_grad: bool, optional
        """
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, data):
        """
        Move input data to the specified device.

        :param data: Input data containing LAB colorspace components.
        :type data: dict
        """
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)

    def forward(self):
        """
        Forward pass through the generator
        """
        self.fake_color = self.net_G(self.L)

    def backward_D(self):
        """
        Backward pass and optimization for the discriminator.
        """
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_predictions = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GAN_criterion(fake_predictions, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_predictions = self.net_D(real_image)
        self.loss_D_real = self.GAN_criterion(real_predictions, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """
        Backward pass for the generator.
        """
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_predictions = self.net_D(fake_image)
        self.loss_G_GAN = self.GAN_criterion(fake_predictions, True)
        self.loss_G_L1 = self.L1_criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize(self):
        """
        Optimization of the whole model.

        This method performs a single optimization step for both the generator and the discriminator.
        It includes the forward pass, backward pass, and parameter updates.

        Steps:
        1. Perform the forward pass through the generator.
        2. Set the discriminator to training mode and enable gradients for its parameters.
        3. Zero the gradients of the discriminator optimizer.
        4. Perform the backward pass for the discriminator and update its parameters.
        5. Set the generator to training mode and disable gradients for the discriminator parameters.
        6. Zero the gradients of the generator optimizer.
        """
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()

        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()
