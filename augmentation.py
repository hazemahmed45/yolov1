import torch




class Compose(object):
    def __init__(self,transforms) -> None:
        super(Compose,self).__init__()
        self.transforms=transforms
        return
    def __call__(self, **kwargs):
        img=kwargs['image']
        bboxes=kwargs['bbox']
        for t in self.transforms:
            img,bboxes=t(img),bboxes
        
        return {'image':img,'bbox':bboxes}
    
    
