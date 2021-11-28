import torch
# from torch.utils.data import data
from torchvision.transforms import transforms
from torch.optim import Adam
from tqdm import tqdm
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from model import YOLOv1
from dataset import VocDataset
from utils.iou import intersection_over_union
from utils.nms import nms
from utils.mean_avg_precision import mean_average_precision
from utils.bboxes_utils import get_bboxes,cellboxes_to_boxes
from utils.model_utils import save_checkpoint,load_checkpoint
from utils.visualization_utils import plot_image
from augmentation import Compose
from loss import YoloLoss
from utils.train_utilts import train_loop



seed =123
torch.manual_seed(seed)

LR=2e-5
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE=16
WEIGHT_DECAY=0
EPOCHS=100
NUM_WORKERS=8
PIN_MEMORY=True
LOAD_MODEL=False
LOAD_MODEL_FILE='model.pt'
IMG_DIR='PascalVoc/images'
LABEL_DIR='PascalVoc/labels'
TRAIN_CSV_FILE='PascalVoc/train.csv'
VALID_CSV_FILE='PascalVoc/test.csv'
TRAIN_CSV_FILE='PascalVoc/8examples.csv'
VALID_CSV_FILE='PascalVoc/8examples.csv'

transform=Compose([transforms.Resize((448,448)),transforms.ToTensor()])
def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

def main():
    model=YOLOv1(split_size=7,num_boxes=2,num_classes=20).to(DEVICE)
    optimizer=Adam(model.parameters(),LR,weight_decay=WEIGHT_DECAY)

    if(LOAD_MODEL):
        load_checkpoint(torch.load(LOAD_MODEL_FILE),model,optimizer)
        
    train_dataset=VocDataset(TRAIN_CSV_FILE,IMG_DIR,LABEL_DIR,transform=transform)
    valid_dataset=VocDataset(VALID_CSV_FILE,IMG_DIR,LABEL_DIR,transform=transform)

    train_loader=DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY)
    valid_loader=DataLoader(dataset=valid_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY)

    criterion=YoloLoss()

    for epoch in range(EPOCHS):
        # for x, y in train_loader:
        #    x = x.to(DEVICE)
        #    for idx in range(8):
        #        bboxes = cellboxes_to_boxes(model(x))
        #        bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
        #        plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)

        #    import sys
        #    sys.exit()

        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")

        #if mean_avg_prec > 0.9:
        #    checkpoint = {
        #        "state_dict": model.state_dict(),
        #        "optimizer": optimizer.state_dict(),
        #    }
        #    save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
        #    import time
        #    time.sleep(10)

        train_fn(train_loader, model, optimizer, criterion)
    # for e in range(EPOCHS):
    #     pred_boxes, target_boxes = get_bboxes(
    #         train_loader, model, iou_threshold=0.5, threshold=0.4
    #     )
    #     # print(pred_boxes,target_boxes)
    #     # exit()
    #     mean_avg_prec = mean_average_precision(
    #         pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
    #     )
    #     print(f"Train mAP: {mean_avg_prec}")
        
    #     model.train()
    #     iter_loop=tqdm(enumerate(train_loader),total=len(train_loader))
    #     mean_loss=[]
        
    #     for ii,(batch_imgs,batch_labels) in iter_loop:
    #         optimizer.zero_grad()
    #         batch_imgs,batch_labels=batch_imgs.to(DEVICE),batch_labels.to(DEVICE)
    #         out=model(batch_imgs)
    #         loss=criterion(out,batch_labels)
    #         mean_loss.append(loss.item())
    #         iter_loop.set_description(desc="Train loop e: "+str(e))
    #         iter_loop.set_postfix({'LOSS':sum(mean_loss)/len(mean_loss)})
    #         loss.backward()
    #         optimizer.step()
            
    #     # model.eval()
    #     # iter_loop=tqdm(enumerate(valid_loader),total=len(valid_loader))
    #     # mean_loss=[]
        
    #     # for ii,(batch_imgs,batch_labels) in iter_loop:
    #     #     batch_imgs,batch_labels=batch_imgs.to(DEVICE),batch_labels.to(DEVICE)
    #     #     with torch.no_grad():
    #     #         out=model(batch_imgs)
    #     #     loss=criterion(out,batch_labels)
    #     #     mean_loss.append(loss.item())
    #     #     iter_loop.set_description(desc="Valid loop e: "+str(e))
    #     #     iter_loop.set_postfix({'LOSS':sum(mean_loss)/len(mean_loss)})
        

if(__name__ == '__main__'):
    main()