import torch
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image


def ef_loss(y_true, y_pred):
    sub=torch.sub(y_true,y_pred)
    abs=torch.abs(sub)
    ef_loss = torch.mean(abs)
    return ef_loss
#[bs,2]

def categorical_crossentropy(label,pred):
    loss=0
    for j in range(pred.shape[-1]):
        loss += -label[j]*torch.log(pred[-1][j])
    return loss

# Loss functions
class PerceptualLoss():
    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model

    def __init__(self, loss):
        self.criterion = loss
        self.contentFunc = self.contentFunc()

    def get_loss(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss

