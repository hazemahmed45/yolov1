import torch
from torch import nn
from torch.functional import split
from config import archi_config




class ConvBlock(nn.Module):
    def __init__(self,in_c,out_c,**kwargs) -> None:
        super(ConvBlock,self).__init__()
        self.conv=nn.Conv2d(in_channels=in_c,out_channels=out_c,bias=False,**kwargs)
        self.batchnorm=nn.BatchNorm2d(out_c)
        self.lrelu=nn.LeakyReLU(0.1)
        return 
    
    def forward(self,x):
        
        return self.lrelu(self.batchnorm(self.conv(x)))
    
class YOLOv1(nn.Module):
    def __init__(self,in_c=3,**kwargs) -> None:
        super(YOLOv1,self).__init__()
        self.archi_config=archi_config
        self.in_c=in_c
        self.darknet=self._create_conv_layers(self.archi_config)
        self.fcs=self._create_fcs(**kwargs)
        return 
    
    def forward(self,x):
        x=self.darknet(x)
        return self.fcs(torch.flatten(x,start_dim=1))
    def _create_conv_layers(self,archi_config):
        layers=[]
        in_c=self.in_c
        for layer in archi_config:
            if(isinstance(layer,str)):
                layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
            elif(isinstance(layer,tuple)):
                k,out_c,s,p=layer
                layers.append(ConvBlock(in_c,out_c,kernel_size=k,stride=s,padding=p))
                in_c=out_c
            else:
                repeat_n=layer[-1]
                for _ in range(repeat_n):
                    for i in range(len(layer)-1):
                        l=layer[i]
                        k,out_c,s,p=l
                        layers.append(ConvBlock(in_c,out_c,kernel_size=k,stride=s,padding=p))
                        in_c=out_c
                # pass
        return nn.Sequential(*layers)
    def _create_fcs(self,split_size,num_boxes,num_classes):
        S,B,C=split_size,num_boxes,num_classes
        
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*S*S,496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496,S*S*(C+B*5))
        )
        
        
def test(split_size=7,num_boxes=2,num_classes=20):
    model=YOLOv1(split_size=split_size,num_boxes=num_boxes,num_classes=num_classes)
    x=torch.randn((2,3,448,448))
    out=model(x)
    print(out.shape)
    print(out.view(-1,split_size,split_size,num_boxes*5+num_classes).shape)
    
if(__name__ =='__main__'):
    test()