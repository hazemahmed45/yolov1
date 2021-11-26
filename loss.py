import torch
from torch import nn
from utils.iou import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self,S=7,B=2,C=20) -> None:
        super(YoloLoss).__init__()
        self.mse=nn.MSELoss(reduce='sum')
        self.S=S
        self.B=B
        self.C=C
        self.lambda_noobj=0.5
        self.lambda_coord=5
        
        return 

    def forward(self,pred,target):
        pred=pred.view((-1,self.S,self.S,self.C+self.B*5))
         
        iou_anch_1=intersection_over_union(pred[...,21:25],target[...,21:25])
        iou_anch_2=intersection_over_union(pred[...,26:30],target[...,21:25])
        ious=torch.cat([iou_anch_1.unsqueeze(0),iou_anch_2.unsqueeze(0)],dim=0)
        iou_maxes,iou_argmax=torch.max(ious,dim=0)
        exists_box=target[...,20].unsqueeze(3) # is there an object
        
        
        # ==================== #
        #  Loss For Box Coords #
        # ==================== #
        box_preds= exists_box * (
            iou_argmax * pred[...,26:30]+ (1-iou_argmax * pred[...,21:25])
        )
        box_targets=exists_box*target[...,21:25]
        
        box_preds[...,2:4]=torch.sign(box_preds[...,2:4]) * torch.sqrt(torch.abs(box_preds[...,2:4]+1e-6))
        
        box_targets=torch.sqrt(box_targets[...,2:4])
        
        
        box_loss=self.mse(torch.flatten(box_preds,end_dim=-2),torch.flatten(box_targets,end_dim=-2))
        
        # ==================== #
        #    Loss For Object   #
        # ==================== #
        obj_pred_box=exists_box*(iou_argmax*pred[...,25:26] +(1-iou_argmax)*pred[...,20:21])
        obj_target_box=exists_box*target[...,20:21]
        object_loss=self.mse(torch.flatten(obj_pred_box),torch.flatten(obj_target_box))
        
        # ==================== #
        #  Loss For No Object  #
        # ==================== #
        no_obj_pred_box_1=(1-exists_box)*pred[...,20:21] 
        no_obj_pred_box_2=(1-exists_box)*pred[...,25:26] 
        no_obj_target_box=(1-exists_box)*target[...,20:21]
        no_object_loss=self.mse(
            torch.flatten(no_obj_pred_box_1,start_dim=1),
            torch.flatten(no_obj_target_box,start_dim=1)
        )+self.mse(
            torch.flatten(no_obj_pred_box_2,start_dim=1),
            torch.flatten(no_obj_target_box,start_dim=1)
        )
        
        # ==================== #
        #    Loss For Class    #
        # ==================== #
        pred_class=exists_box*pred[...,:20]
        target_class=exists_box*target[...,20]
        class_loss=self.mse(
            torch.flatten(pred_class,end_dim=-2),
            torch.flatten(target_class,end_dim=-2)
        )
        loss=(self.lambda_coord*box_loss)+object_loss+(self.lambda_noobj*no_object_loss)+class_loss
        return loss