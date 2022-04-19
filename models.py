import torch



class encoder_block(torch.nn.Module):
    def __init__(self,input_channel,down_filter):
        super(encoder_block, self).__init__()
        self.LeakyReLU = torch.nn.LeakyReLU(0.2)
        self.Conv_3_2 = torch.nn.Conv2d(in_channels=input_channel, out_channels=down_filter, kernel_size=(3, 3), stride=2, padding=1)
        self.BatchNorm2d = torch.nn.BatchNorm2d(down_filter)
    def forward(self,X):
        X = self.Conv_3_2(X)
        X = self.BatchNorm2d(X)
        X = self.LeakyReLU(X)
        return X

class SeparableConv2D(torch.nn.Module):
    def __init__(self,filters=256,dilation_r=1,padding=0):
        super(SeparableConv2D, self).__init__()
        self.depth_conv = torch.nn.Conv2d(in_channels=filters,out_channels=filters,
                                    kernel_size=(3,3),groups=filters,padding=padding,dilation=dilation_r)
        self.point_conv = torch.nn.Conv2d(in_channels=filters,out_channels=filters,
                                    kernel_size=(1,1),dilation=dilation_r)

    def forward(self,input):

        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class novel_residual_block(torch.nn.Module):
    def __init__(self,filters):
        super(novel_residual_block, self).__init__()
        self.ReflectionPad = torch.nn.ReflectionPad2d(1)
        self.ReflectionPad2 = torch.nn.ReflectionPad2d(2)
        self.LeakyReLU = torch.nn.LeakyReLU(0.2)

        self.SeparableConv2D_1 = SeparableConv2D(filters=filters, dilation_r=1)
        self.SeparableConv2D_2 = SeparableConv2D(filters=filters, dilation_r=1)
        self.SeparableConv2D_3 = SeparableConv2D(filters=filters, dilation_r=2)

        self.BatchNorm2d = torch.nn.BatchNorm2d(filters)
    def forward(self,X_input):
        X = X_input
        X = self.ReflectionPad(X)
        X = self.SeparableConv2D_1(X)
        X = self.BatchNorm2d(X)
        X = self.LeakyReLU(X)

        X_branch_1 = self.ReflectionPad(X)
        X_branch_1 = self.SeparableConv2D_2(X_branch_1)
        X_branch_1 = self.BatchNorm2d(X_branch_1)
        X_branch_1 = self.LeakyReLU(X_branch_1)

        ## Branch 2
        X_branch_2 = self.ReflectionPad2(X)
        X_branch_2 = self.SeparableConv2D_3(X_branch_2)
        X_branch_2 = self.BatchNorm2d(X_branch_2)
        X_branch_2 = self.LeakyReLU(X_branch_2)
        X_add_branch_1_2 = torch.add(X_branch_2, X_branch_1)
        X = torch.add(X_input, X_add_branch_1_2)
        return X


class decoder_block(torch.nn.Module):
    def __init__(self,input,filter):
        super(decoder_block, self).__init__()
        self.convT = torch.nn.ConvTranspose2d(in_channels=input, out_channels=filter, kernel_size=(3, 3), stride=2, padding=1,output_padding=1)
        self.BatchNorm2d = torch.nn.BatchNorm2d(filter)
        self.LeakyReLU = torch.nn.LeakyReLU(0.2)
    def forward(self,X):
        X = self.convT(X)
        X = self.BatchNorm2d(X)
        X = self.LeakyReLU(X)
        return X

class Attention(torch.nn.Module):
    def __init__(self,input,filters):
        super(Attention, self).__init__()
        self.LeakyReLU = torch.nn.LeakyReLU(0.2)
        self.BatchNorm = torch.nn.BatchNorm2d(filters)
        self.Conv_3_1 = torch.nn.Conv2d(in_channels=input, out_channels=filters, kernel_size=(3, 3), padding=1)
    def forward(self,X):
        X_input = X
        X = self.Conv_3_1(X)
        X = self.BatchNorm(X)
        X = self.LeakyReLU(X)
        X = torch.add(X_input, X)
        X = self.Conv_3_1(X)
        X = self.BatchNorm(X)
        X = self.LeakyReLU(X)
        X = torch.add(X_input, X)
        return X