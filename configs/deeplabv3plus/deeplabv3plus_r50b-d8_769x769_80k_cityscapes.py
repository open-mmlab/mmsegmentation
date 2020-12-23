_base_ = './deeplabv3plus_r50-d8_769x769_80k_cityscapes.py'
model = dict(type='ResNet', pretrained='torchvision://resnet50')
