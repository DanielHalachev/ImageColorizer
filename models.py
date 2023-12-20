import torch
from torch import nn, optim
from loss import GANLoss


class UnetBlock(nn.Module):
    def __init__(self, nf, ni, submodule=None, input_channels=None, dropout=False, innermost=False, outermost=False):
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
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class Unet(nn.Module):
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
        return self.model(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, nfd=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, nfd, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(nfd, nfd * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nfd * 2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(nfd * 2, nfd * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nfd * 4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(nfd * 4, nfd * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(nfd * 8),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(nfd * 8, nfd * 16, 4, 1, 1, bias=True),
        )

    def forward(self, x):
        return self.model(x)


def init_weights(net, init='norm', gain=0.02):
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
    model = model.to(device)
    model = init_weights(model)
    return model


class MainModel(nn.Module):
    def __init__(self,
                 net_generator=None,
                 lr_generator=2e-4,
                 lr_discriminator=2e-4,
                 beta1=0.5,
                 beta2=0.999,
                 lambda_l1=100.):
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
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)

    def forward(self):
        self.fake_color = self.net_G(self.L)

    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_predictions = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GAN_criterion(fake_predictions, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_predictions = self.net_D(real_image)
        self.loss_D_real = self.GAN_criterion(real_predictions, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_predictions = self.net_D(fake_image)
        self.loss_G_GAN = self.GAN_criterion(fake_predictions, True)
        self.loss_G_L1 = self.L1_criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize(self):
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
