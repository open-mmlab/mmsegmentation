import torch
from collections import OrderedDict
import argparse

def ddrnet_convert(old_path,new_path,net_type):

        # keys change
    if net_type=='DDRNet23' or   net_type=='DDRNet23s' :
        old=['conv1.','layer1.','layer2.','layer3.','layer4.','compression3.','compression4.','down3.','down4.','layer3_.','layer4_.','layer5_.','layer5.' ]
        new=['conv1.','layer1.','layer2.','layer3_fuison.main_layer.','layer4_fuison.main_layer.','layer3_fuison.compression.',\
         'layer4_fuison.compression.','layer3_fuison.down.',\
         'layer4_fuison.down.','layer3_fuison.sub_layer.','layer4_fuison.sub_layer.','layer5_.','layer5.']
    elif net_type=='DDRNet39' :
        old=['conv1.','layer1.','layer2.','layer3_1.','layer4.','compression3_1.','compression4.',\
             'down3_1.','down4.','layer3_1_.','layer4_.','layer5_.','layer5.',\
             'layer3_2.','layer3_2_.','compression3_2.','down3_2.']
        new=['conv1.','layer1.','layer2.','layer3_fuison.main_layer.','layer4_fuison.main_layer.','layer3_fuison.compression.',\
             'layer4_fuison.compression.','layer3_fuison.down.',\
             'layer4_fuison.down.','layer3_fuison.sub_layer.','layer4_fuison.sub_layer.','layer5_.','layer5.',  \
             'layer3_fuison_2.main_layer.','layer3_fuison_2.sub_layer.','layer3_fuison_2.compression.','layer3_fuison_2.down.']
                
    
    old_model=torch.load(old_path,map_location='cpu')
    keys_old_model=list (  old_model.keys() )
   
    new_model=OrderedDict()

    for v in range(len(keys_old_model)) :
        r=keys_old_model[v] 
        for i in range(len(old)):
            if  r.startswith(old[i]):
                t=r.replace(old[i],new[i])
                new_model[t]=old_model[r]
    torch.save(new_model,new_path)

def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in official pretrained DDRNet23/DDRNet23s/DDRNet39 to '
        'MMSegmentation style.')
    parser.add_argument('--src', default='DDRNet23s_imagenet.pth',help='src model path')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('--dst', default='DDRNet23s_imagenet_mmseg.pth',help='save path')
    parser.add_argument('--type', help='model type: DDRNet23s or DDRNet23 or DDRNet39')
    args = parser.parse_args()
    
    old_path=args.src
    new_path=args.dst
    ddrnet_convert(old_path,new_path,args.type)


if __name__ == '__main__':
    main()
