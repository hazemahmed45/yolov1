import os
import pandas as pd
from PIL import Image
from torch.utils import data
from torch.utils.data import Dataset
import torch
import torchvision

class VocDataset(Dataset):
    def __init__(self,csv_file,img_dir,label_dir,S=7,B=2,C=20,transform=None):
        super(VocDataset,self).__init__()
        self.dataframe=pd.read_csv(csv_file)
        self.img_dir=img_dir
        self.label_dir=label_dir
        self.transform=transform
        self.S=S
        self.C=C
        self.B=B
        return 
    
    def __getitem__(self, index) :
        img_filename,label_filename=self.dataframe.iloc[index]
        img_path=os.path.join(self.img_dir,img_filename)
        label_path=os.path.join(self.label_dir,label_filename)
        img=Image.open(img_path)
        bboxes=self.read_bboxes_from_file(label_path)
        
        bboxes=torch.tensor(bboxes)
        if(self.transform is not None):
            transforms_dict=self.transform(image=img,bbox=bboxes)
            img=transforms_dict['image']
            bboxes=transforms_dict['bbox']
        else:
            img=torchvision.transforms.ToTensor()(img)
        
        label_matrix=torch.zeros((self.S,self.S,self.C+5))
        for bbox in bboxes:
            class_label,x,y,w,h = bbox
            class_label=int(class_label)
            i,j=int(y*self.S),int(self.S*x)
            x_cell,y_cell=self.S*x-j,self.S*y-i
            w_cell,h_cell=self.S*w,self.S*h

            if(label_matrix[i,j,20]==0):
                label_matrix[i,j,20]=1
                label_matrix[i,j,21:25]=torch.tensor([x_cell,y_cell,w_cell,h_cell])
                label_matrix[i,j,class_label]=1
        
        
        return img ,label_matrix 
    
    
    def __len__(self):
        
        return len(self.dataframe.shape[0])
    def read_bboxes_from_file(self,filepath):
        bboxes=[]
        with open(filepath,'r') as f:
            for l in f.readlines():
                class_label,center_x,center_y,w,h=l.replace("\n","").split()
                bboxes.append([int(class_label),float(center_x),float(center_y),float(w),float(h)])
            
        return bboxes
    
    
if __name__ == '__main__':
    dataset=VocDataset('PascalVoc/train.csv','PascalVoc/images','PascalVoc/labels')
    img,label=dataset[0]
    print(img.shape,label.shape)
    pass