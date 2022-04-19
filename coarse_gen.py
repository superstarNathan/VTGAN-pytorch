import torch
from models import encoder_block,novel_residual_block,Attention,decoder_block


class coarse_generator(torch.nn.Module):
    def __init__(self,ncf=64, n_downsampling=2, n_blocks=9):
        super(coarse_generator, self).__init__()
        self.ncf=ncf
        self.n_blocks = n_blocks
        self.n_downsampling=n_downsampling

        self.Conv_1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), padding=0)
        self.Conv_2 = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(7, 7), padding=0)

        self.BatchNorm2d_64 = torch.nn.BatchNorm2d(64)
        self.LeakyReLU = torch.nn.LeakyReLU(0.2)

        self.ReflectionPad3 = torch.nn.ReflectionPad2d(3)

        up_filters = int(self.ncf * pow(2, (self.n_downsampling - 0)) / 2)
        self.decoder_block1 = decoder_block(256, up_filters)

        up_filters_2 = int(self.ncf * pow(2, (self.n_downsampling - 1)) / 2)
        self.decoder_block2 = decoder_block(128, up_filters_2)

        self.Attention1 = Attention(128, 128)
        self.Attention2 = Attention(64, 64)

        down_filters_1 = 64 * pow(2, 0) * 2
        self.encoder_block1 = encoder_block(64, down_filters_1)

        down_filters_2 = 64 * pow(2, 1) * 2
        self.encoder_block2 = encoder_block(128, down_filters_2)

        self.novel_residual_block1 = novel_residual_block(filters=256)

    def forward(self,X_input):
        X = self.ReflectionPad3(X_input)
        X = self.Conv_1(X)
        X = self.BatchNorm2d_64(X)
        X_pre_down = self.LeakyReLU(X)

        # Downsampling layers
        X_down1 = self.encoder_block1(X)
        X_down2 = self.encoder_block2(X_down1)
        X = X_down2
        # ====================================================================
        # Novel Residual Blocks
        res_filters = pow(2, self.n_downsampling)
        for i in range(self.n_blocks):
            X = self.novel_residual_block1(X)

        # Upsampling layers

        X_up1 = self.decoder_block1(X)
        X_up1_att = self.Attention1(X_down1)
        X_up1_add = torch.add(X_up1_att, X_up1)

        X_up2 = self.decoder_block2(X_up1_add)
        X_up2_att = self.Attention2(X_pre_down)
        X_up2_add = torch.add(X_up2_att, X_up2)
        feature_out = X_up2_add

        X = self.ReflectionPad3(X_up2_add)
        X =self.Conv_2(X)
        X = torch.tanh(X)
        return X, feature_out


