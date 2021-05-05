import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.utils.spectral_norm(nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False)),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(GeneratorUNet, self).__init__()
                                                                                # (256,256)(n/a,1) (starting dimensions of images)(w,h)(input_ch, output_ch)
        self.down1 = UNetDown(in_channels, 64, normalize=False)                 # (128,128)(1,64)
        self.down2 = UNetDown(64, 128)                                          # (64,64)(64,128)
        self.down3 = UNetDown(128, 256)                                         # (32,32)(128,256)
        self.down4 = UNetDown(256, 512, dropout=0.5)                            # (16,16)(256,512)
        self.down5 = UNetDown(512, 512, dropout=0.5)                            # (8,8)(512,512)
        self.down6 = UNetDown(512, 512, dropout=0.5)                            # (4,4)(512,512)
        self.down7 = UNetDown(512, 512, dropout=0.5)                            # (2,2)(512,512)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)           # (1,1)(512,512)

        self.up1 = UNetUp(512, 512, dropout=0.5)                                # (2,2)(512,512)
        self.up2 = UNetUp(1024, 512, dropout=0.5)                               # (4,4)(1024,512)
        self.up3 = UNetUp(1024, 512, dropout=0.5)                               # (8,8)(1024,512)
        self.up4 = UNetUp(1024, 512, dropout=0.5)                               # (16,16)(1024,512)
        self.up5 = UNetUp(1024, 256)                                            # (32,32)(1024,256)
        self.up6 = UNetUp(512, 128)                                             # (64,64)(512,128)
        self.up7 = UNetUp(256, 64)                                              # (128,128)(256,64)

        self.final = nn.Sequential(                                             # (256,256)(128,1)
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, act_dim=1):
        super(Discriminator, self).__init__()
        self.act_dim = act_dim

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            if normalization:
                layers = [nn.utils.spectral_norm(nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1))]
            else:
                layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]

            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.cnn = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512)
        )

        self.disc = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

        self.flatten = nn.Sequential(
            nn.Flatten()
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.flatten(self.cnn(torch.zeros((1,2,256,256)).float())).shape[1]

        # print(n_flatten.shape[1])
        self.act_net =  nn.Sequential(
            nn.Linear(n_flatten, 64), nn.Tanh(),
            nn.Linear(64, act_dim), nn.Tanh()
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        cnn_out = self.cnn(img_input)
        disc_out = self.disc(cnn_out)
        act_out = self.act_net(self.flatten(cnn_out))

        return disc_out, act_out
