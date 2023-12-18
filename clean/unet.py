import torch
from torch import nn


class UNet(nn.Module):
    def __init__(self, ngf):
        super().__init__()

        # Encoder (contracting path)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(ngf),

            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(ngf * 2),

            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(ngf * 4),

            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(ngf * 8),

            nn.Conv2d(ngf * 8, ngf * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(ngf * 16),

            nn.Conv2d(ngf * 16, ngf * 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(ngf * 32),

            nn.Conv2d(ngf * 32, ngf * 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(ngf * 64),

            nn.Conv2d(ngf * 64, ngf * 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm2d(ngf * 128)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ngf * 128, ngf * 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(ngf * 256)
        )

        # Decoder (expanding path)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ngf * 256, ngf * 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(ngf * 128),

            nn.ConvTranspose2d(ngf * 128, ngf * 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(ngf * 64),

            nn.ConvTranspose2d(ngf * 64, ngf * 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(ngf * 32),

            nn.ConvTranspose2d(ngf * 32, ngf * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(ngf * 16),

            nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(ngf * 8),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(ngf * 4),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(ngf * 2),

            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(ngf),
        )
        self.final = nn.ConvTranspose2d(ngf, 2, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        enc_1 = self.encoder[0:3](x)
        enc_2 = self.encoder[3:6](enc_1)
        enc_3 = self.encoder[6:9](enc_2)
        enc_4 = self.encoder[9:12](enc_3)
        enc_5 = self.encoder[12:15](enc_4)
        enc_6 = self.encoder[15:18](enc_5)
        enc_7 = self.encoder[18:21](enc_6)
        enc_8 = self.encoder[21:24](enc_7)

        # Bottleneck
        bottleneck = self.bottleneck(enc_8)

        # Decoder with skip connections
        dec = self.decoder[0:3](bottleneck)
        dec = torch.cat((dec, enc_8), dim=1)
        dec = self.decoder[3:6](dec)
        dec = torch.cat((dec, enc_7), dim=1)
        dec = self.decoder[6:9](dec)
        dec = torch.cat((dec, enc_6), dim=1)
        dec = self.decoder[9:12](dec)
        dec = torch.cat((dec, enc_5), dim=1)
        dec = self.decoder[12:15](dec)
        dec = torch.cat((dec, enc_4), dim=1)
        dec = self.decoder[15:18](dec)
        dec = torch.cat((dec, enc_3), dim=1)
        dec = self.decoder[18:21](dec)
        dec = torch.cat((dec, enc_2), dim=1)
        dec = self.decoder[21:24](dec)
        dec = torch.cat((dec, enc_1), dim=1)

        # TODO

        final_output = torch.sigmoid(self.final(dec))

        return final_output
