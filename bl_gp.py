import torch
import torchvision.models as models
resnext = models.resnext50_32x4d(pretrained=False)
a = models.alexnet(pretrained=False)
vgg = models.vgg16(pretrained=False)
vgg_2 = models.vgg16(pretrained=False)
vgg_3 = models.vgg16(pretrained=False)
vgg_5 = models.vgg16(pretrained=False)
vgg_4 = models.vgg16(pretrained=False)
vgg_6 = models.vgg16(pretrained=False)
vgg_7 = models.vgg16(pretrained=False)

# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
resnext.train()
resnext.to('cuda:1')
a.train()
a.to('cuda:1')
vgg.train()
vgg.to('cuda:1')

vgg_4.train()
vgg_4.to('cuda:1')

vgg_2.train()
vgg_2.to('cuda:1')

vgg_3.train()
vgg_3.to('cuda:1')
vgg_5.train()
vgg_5.to('cuda:1')


vgg_6.train()
vgg_6.to('cuda:1')

vgg_7.train()
vgg_7.to('cuda:1')


while(1):
    pass
