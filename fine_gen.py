import torch
from models import encoder_block,novel_residual_block,Attention,decoder_block,SeparableConv2D

class fine_generator(torch.nn.Module):
    def __init__(self, nff=64,n_coarse_gen=1):
        super(fine_generator, self).__init__()
        self.LeakyReLU = torch.nn.LeakyReLU(0.2)
        self.nff=nff
        self.n_coarse_gen = n_coarse_gen

        self.Conv_7_1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), padding=0)
        self.Conv_7_1_2 = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(7, 7), padding=0)

        self.ReflectionPad3 = torch.nn.ReflectionPad2d(3)

        self.BatchNorm2d_64 = torch.nn.BatchNorm2d(64)
        self.BatchNorm2d_128 = torch.nn.BatchNorm2d(128)

        self.encoder_block1 = encoder_block(64, 64)
        self.SeparableConv2D_1 = SeparableConv2D(filters=128, dilation_r=1, padding=1)
        self.novel_residual_block1 = novel_residual_block(128)
        self.decoder_block1 = decoder_block(128, 64)
        self.Attention1 = Attention(64, 64)

    def forward(self,X_input, X_coarse):
        # Downsampling layers
        X = self.ReflectionPad3(X_input)
        X = self.Conv_7_1(X)
        X = self.BatchNorm2d_64(X)
        X_pre_down = self.LeakyReLU(X)

        X_down1 = self.encoder_block1(X)
        print("X_down1.shape",X_down1.shape)
        print("X_coarse.shape",  X_coarse.shape)
        # X_down1.shape
        # torch.Size([4, 64, 256, 256])
        # X_coarse.shape
        # torch.Size([4, 64, 256, 256])
        X= torch.concat([X_coarse, X_down1],1)
        X = self.SeparableConv2D_1(X)

        X = self.BatchNorm2d_128(X)
        X = self.LeakyReLU(X)
        for j in range(2):
            X = self.novel_residual_block1(X)
        X_up1 = self.decoder_block1(X)
        X_up1_att = self.Attention1(X_pre_down)
        X_up1_add = torch.add(X_up1_att, X_up1)
        X = self.ReflectionPad3(X_up1_add)
        X = self.Conv_7_1_2(X)
        X = torch.tanh(X)
        return X


