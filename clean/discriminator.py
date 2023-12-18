from torch import nn


class Discriminator(nn.Module):
    def __init__(self, ngpu, ndf, nc, patch_size, num_patches):
        """
        Initialize the Discriminator of the GAN Network
        :param ngpu: Number of GPUs to use
        :param ndf: Number of feature maps in discriminator
        :param nc: Number of color channels
        """
        super(Discriminator, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.ngpu = ngpu

        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(-1, x.shape[1], self.patch_size, self.patch_size)
        patch_scores = self.main(patches)
        patch_scores = patch_scores.view(-1, self.num_patches)
        return self.main(x)
