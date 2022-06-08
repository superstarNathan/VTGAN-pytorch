from dataloader import *
import torch.nn.functional as F
import torch
import torch.optim as optim
from einops import rearrange

class PatchEncoder(torch.nn.Module):
    def __init__(self, num_patches=64, projection_dim=64):
        super(PatchEncoder, self).__init__()
        self.num_patches= num_patches
        self.projection = torch.nn.Linear(64*64*4,out_features=projection_dim)#256*256:262144
        self.projection_resize = torch.nn.Linear(32*32*4, out_features=projection_dim)
        #self.position_embedding = torch.nn.Embedding(num_patches,projection_dim)

    def forward(self, input):
        # positions = torch.arange(0,self.num_patches)
        # positions=positions.unsqueeze(0).to(torch.device("cuda:0"))
        #positions = torch.nn.Parameter(torch.randn(1, self.num_patches, self.num_patches)).to(torch.device("cuda:0"))
        positions = torch.nn.Parameter(torch.randn(1, self.num_patches, self.num_patches)).to(torch.device("cpu"))
        print(positions.shape)
        print(input.shape)
        if input.shape[-1]==4096:
            encoded = self.projection_resize(input)+positions
        else:
             encoded = self.projection(input) + positions
        return encoded
        # if input.shape[-1]==4096:
        #     encoded = self.projection_resize(input)+self.position_embedding(positions)
        # else:
        #      encoded = self.projection(input) + self.position_embedding(positions)
        # return encoded

class vit_discriminator(torch.nn.Module):
    def __init__(self,):
        super(vit_discriminator, self).__init__()
        self.GELU = torch.nn.GELU()
        self.Conv_4_1 = torch.nn.Conv2d(1, 1, (4, 4), padding='same')
        self.Conv_4_1_2 = torch.nn.Conv2d(1, 64, (4,4), padding='same')
        self.MultiHeadAttention = torch.nn.MultiheadAttention(64, 4, dropout=0.1).cuda()
        #self.LayerNorm = torch.nn.LayerNorm(64, eps=1e-6)
        self.linear1 = torch.nn.Linear(64, 64 ** 2)
        self.linear2 = torch.nn.Linear(64 ** 2, 64)
        self.linear3 = torch.nn.Linear(64, 2)
        self.Softmax = torch.nn.Softmax()
        self.AdaptiveAvgPool2d = torch.nn.AdaptiveAvgPool2d(1)
        self.num_patches = (512 // 64) ** 2
        self.MSELoss = torch.nn.MSELoss()
        self.PatchEncoder=PatchEncoder()
        self.LayerNormalization_0 = torch.nn.LayerNorm(normalized_shape=[64, 64], eps=1e-6)
        self.LayerNormalization_1 = torch.nn.LayerNorm(normalized_shape=[64, 64], eps=1e-6)
        self.LayerNormalization_2 = torch.nn.LayerNorm(normalized_shape=[64, 64], eps=1e-6)

    def mlp(self,x,dropout_rate):
        x = self.linear1(x)
        x = self.GELU(x)
        x = F.dropout(x, p=dropout_rate)
        x = self.linear2(x)
        x = self.GELU(x)
        x = F.dropout(x, p=dropout_rate)
        return x

    def forward(self,fundus,angio, patch_size, transformer_layers=8):
        X = torch.concat((fundus, angio), 1)
        feat = []
        patches = rearrange(X, 'b c (h h1) (w w2) -> b (h w) (h1 w2 c)', h1=patch_size, w2=patch_size)
        print(patches.shape)
        encoded_patches = self.PatchEncoder(patches)
        for _ in range(transformer_layers):
            x1 = self.LayerNormalization_1(encoded_patches)
            if x1.ndim != 3:
                x1 = x1.unsqueeze(0)
            attention_output, attn_output_weights = self.MultiHeadAttention(x1, x1, x1)
            x2 = torch.add(attention_output, encoded_patches)
            x3 = self.LayerNormalization_2(x2)
            x3 = self.mlp(x3, dropout_rate=0.1)
            encoded_patches = torch.add(x3, x2)
            feat.append(encoded_patches)
        feat = torch.concat([feat[0], feat[1], feat[2], feat[3]], -1)

        representation = self.LayerNormalization_0(encoded_patches)
        X_reshape = representation.unsqueeze(0)
        X = self.Conv_4_1(X_reshape)
        out_hinge = torch.tanh(X)
        representation = self.Conv_4_1_2(X_reshape)
        features = self.AdaptiveAvgPool2d(representation).squeeze(-1).squeeze(-1)
        classses = self.linear3(features)
        out_class = self.Softmax(classses)
        return out_hinge, out_class, feat


device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# #
# #
# # nlr=0.0002
# # nbeta1=0.5
# #
from loss_torch import categorical_crossentropy
# #
# #vit_discriminator = torch.nn.DataParallel(vit_discriminator.to(device))
# vit_discriminator.zero_grad()
# #ptimizerD = optim.Adam(vit_discriminator.parameters(), lr=nlr, betas=(nbeta1, 0.999))
# out_hinge, out_class, feat=vit_discriminator(next(fundus_loader_resize).to(device),next(angio_loader_resize).to(device),patch_size=32)
# # print("done")
# print(out_class)
# print(out_class.shape)
# nor_label=[0,1]
# ca=categorical_crossentropy(nor_label,out_class)
# print(ca)

#loss function
# categorical_crossentropy_loss = -(0 * torch.log(torch.tensor(output)) + 0.9 * torch.log(torch.tensor(1-output)))
# ef_loss=ef_loss(label,)
# MSE=torch.nn.MSELoss()
# MSE=MSE()
# loss=categorical_crossentropy_loss+ef_loss+MSE
# loss.backward()

# print(out_hinge.shape)
# print(out_class.shape)
# print(feat.shape)

# torch.Size([1, 1, 64, 64])
# torch.Size([1, 2])
# torch.Size([1, 64, 256])