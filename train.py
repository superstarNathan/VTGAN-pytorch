import torch.optim as optim
import torch
from dataloader import *
from newvit import vit_discriminator
from fine_gen import fine_generator
from coarse_gen import coarse_generator
from loss_torch import ef_loss,categorical_crossentropy,PerceptualLoss,Hinge_Loss
import matplotlib.pyplot as plt
import random
from torchvision import transforms

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nlr=0.0002
nbeta1=0.5

fine_discriminator=vit_discriminator()
fine_discriminator.to(device)
optimizerD_f = optim.Adam(fine_discriminator.parameters(), lr=nlr, betas=(nbeta1, 0.999))

coarse_discriminator=vit_discriminator()
coarse_discriminator.to(device)
optimizerD_c = optim.Adam(fine_discriminator.parameters(), lr=nlr, betas=(nbeta1, 0.999))

fine_generator=fine_generator()
fine_generator.to(device)
optimizerG_f = optim.Adam(fine_generator.parameters(), lr=nlr, betas=(nbeta1, 0.999))

coarse_generator=coarse_generator()
coarse_generator.to(device)
optimizerG_c = optim.Adam(coarse_generator.parameters(), lr=nlr, betas=(nbeta1, 0.999))

MSELoss=torch.nn.MSELoss()
perceptual_loss = PerceptualLoss(torch.nn.MSELoss())
Hinge_Loss=Hinge_Loss().cuda()
BCELoss=torch.nn.BCELoss()
L1loss=torch.nn.L1Loss()

# generate a batch of fake samples for Coarse Generator
#generatelabel
epoches=100

label_true= torch.ones(1,1,64,64).to(device)
label_false=-1*torch.ones(1,1,64,64).to(device)

nor_label=[0,0.9]
ab_label=[0.9,0]

Dloss1=[]
Dloss2=[]
Dloss3=[]
Dloss4=[]

Dloss5=[]
Dloss6=[]
Dloss7=[]
Dloss8=[]

Gloss1=[]
Gloss2=[]

nor_fundus_resize=next(fundus_loader_resize).to(device)
nor_fundus=next(fundus_loader).to(device)
nor_angio_resize=next(angio_loader_resize).to(device)
nor_angio=next(angio_loader).to(device)

ab_fundus_resize=next(ab_fundus_loader_resize).to(device)
ab_fundus=next(ab_fundus_loader).to(device)
ab_angio_resize=next(ab_angio_loader_resize).to(device)
ab_angio=next(ab_angio_loader).to(device)


for epoch in range(epoches):
    datachoice = [0, 1]
    ret = random.choice(datachoice)
    if ret==1:
        fundus = nor_fundus
        fundus_resize = nor_fundus_resize
        angio = nor_angio
        angio_resize = nor_angio_resize
        label = nor_label
    else:
        fundus = ab_fundus
        fundus_resize = ab_fundus_resize
        angio = ab_angio
        angio_resize = ab_angio_resize
        label = ab_label

    torch.cuda.empty_cache()
    coarse_image, feature_out=coarse_generator(fundus_resize)
    # generate a batch of fake samples for Fine Generator
    fine_image=fine_generator(fundus,feature_out)
    print(fine_image.shape)
    for n_D in range(2):
        print("==============")
        ## FINE DISCRIMINATOR
        # update discriminator for real samples
        optimizerD_f.zero_grad()
        fine_real_hinge, fine_real_class, fine_real_feat=fine_discriminator(fundus,angio,patch_size=64)
        print(fine_real_hinge.shape)
        ca_loss1=categorical_crossentropy(label,fine_real_class)
        hinge_loss1=Hinge_Loss(fine_real_hinge,label_true)
        dloss1=10*hinge_loss1+10*ca_loss1
        dloss1.backward()
        optimizerD_f.step()

        # update discriminator for generated samples
        fine_fake_hinge, fine_fake_class, fine_fake_feat=fine_discriminator(fundus,fine_image.detach(),patch_size=64)
        #MSE_loss2=MSELoss(fine_fake_hinge,label_false)
        hinge_loss2 = Hinge_Loss(fine_fake_hinge,label_false)
        ef_loss1=ef_loss(fine_fake_feat,fine_real_feat.detach())
        ca_loss2=categorical_crossentropy(label,fine_fake_class)
        dloss2=10*hinge_loss2+10*ca_loss2+ef_loss1
        dloss2.backward()
        optimizerD_f.step()

        ## COARSE DISCRIMINATOR
        # update discriminator for real samples#定义了两个判别器
        optimizerD_c.zero_grad()
        coarse_real_hinge, coarse_real_class, coarse_real_feat=coarse_discriminator(fundus_resize,angio_resize,patch_size=32)
        hinge_loss3=Hinge_Loss(coarse_real_hinge,label_true)
        ca_loss3=categorical_crossentropy(label,coarse_real_class)
        dloss3=10*hinge_loss3+10*ca_loss3
        dloss3.backward()
        optimizerD_c.step()

        # update discriminator for generated samples
        coarse_fake_hinge, coarse_fake_class, coarse_fake_feat=coarse_discriminator(fundus_resize,coarse_image.detach(),patch_size=32)
        ef_loss2 = ef_loss(coarse_fake_feat,coarse_real_feat.detach())
        ca_loss4 = categorical_crossentropy(label,coarse_fake_class)
        hinge_loss4 = Hinge_Loss(coarse_fake_hinge,label_false)
        dloss4 = 10*hinge_loss4+10*ca_loss4+ef_loss2
        dloss4.backward()
        optimizerD_c.step()

    Dloss1.append(dloss1)
    Dloss2.append(dloss2)
    Dloss3.append(dloss3)
    Dloss4.append(dloss4)
    err_d = dloss1 + dloss2 + dloss3 + dloss4

    optimizerG_c.zero_grad()
    coarse_image_resize, feature_out=coarse_generator(fundus_resize)

    g_vit1,_,_=coarse_discriminator(fundus_resize,coarse_image_resize,patch_size=32)
    MSE_loss5=MSELoss(coarse_image_resize,angio_resize)
    coarse_Perceptual=torch.concat([coarse_image_resize,coarse_image_resize,coarse_image_resize],1)
    angio_resize_Perceptual=torch.concat([angio_resize,angio_resize,angio_resize],1)
    loss_Perceptual1 = perceptual_loss.get_loss(coarse_Perceptual,angio_resize_Perceptual)
    #gloss_hinge1=Hinge_Loss(g_vit1,label_true)
    gl1=L1loss(g_vit1,label_true)
    gloss1=10*MSE_loss5+10*loss_Perceptual1+10*gl1
           #+10*gloss_hinge1
    Gloss1.append(gloss1)
    gloss1.backward()
    optimizerG_c.step()

    # update the Fine generator
    optimizerG_f.zero_grad()
    fine_image=fine_generator(fundus,feature_out.detach())
    g_vit2,_,_=fine_discriminator(fine_image)
    MSE_loss6=MSELoss(fine_image,angio)
    fine_Perceptual=torch.concat([fine_image,fine_image,fine_image],1)
    angio_Perceptual=torch.concat([angio,angio,angio],1)
    loss_Perceptual2 = perceptual_loss.get_loss(fine_Perceptual,angio_Perceptual)
    #gloss_hinge2=Hinge_Loss(g_vit2,label_true)
    gl2=BCELoss(g_vit2,label_true)
    gloss2=10*MSE_loss6+10*loss_Perceptual2+10*gl2
           #+10*gloss_hinge2
    Gloss2.append(gloss2)
    gloss2.backward()
    optimizerG_f.step()

    err_g=gloss2+gloss1
    print("GAN: loss d: %.5f    loss g: %.5f" % (err_d, err_g))

torch.save({'finG':fine_generator.state_dict(),'coaG':coarse_generator.state_dict()}, 'params.pth')
plt.figure(figsize=(10, 5))
plt.title("generator and discriminator loss during training")
plt.plot(Gloss1,label="G1")
plt.plot(Gloss2, label="G2")
plt.plot(Dloss1,label="D1")
plt.plot(Dloss2, label="D2")
plt.plot(Dloss3,label="D3")
plt.plot(Dloss4, label="D4")
plt.xlabel("iterations")  # iterations迭代
plt.ylabel("Loss")
plt.legend()
plt.savefig("Loss.png")
plt.show()



for i in range(11):#1000表示有1000张图片
    m=i+20
    tpath=os.path.join(r'E:\PycharmProjects\mynewGAN\data\test\ab\\'+ str(m)+'.jpg')     #路径(/home/ouc/river/test)+图片名（img_m）
    fopen = Image.open(tpath)
    transform_resize = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor()])
    data_resize = transform_resize(fopen)  # data就是预处理后，可以送入模型进行训练的数据了
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()])
    print("===============")
    print(data_resize.shape)
    data = transform(fopen)
    print("===============")
    print(data.shape)
    coarse_image_resize, feature_out=coarse_generator(data_resize.unsqueeze(0).to(device))

    print(feature_out.shape)
    output=fine_generator(data.unsqueeze(0).to(device),feature_out)
    toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
    pic = toPIL(output.squeeze(0))
    pic.save('E:\PycharmProjects\mynewGAN\data\\output\\'+str(i)+'.jpg')
    print("done")
