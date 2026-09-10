# MediScan — End-to-End Training Script
# Run in Google Colab with GPU recommended.
# Dataset: PneumoniaMNIST (MedMNIST)
# Architecture: ResNet50 transfer learning

!pip -q install medmnist grad-cam mlflow seaborn

import os, random, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, roc_curve
from medmnist import PneumoniaMNIST

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path("mediscan_outputs"); OUT.mkdir(exist_ok=True)

train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(8),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
eval_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_ds = PneumoniaMNIST(split="train", transform=train_tf, download=True)
val_ds   = PneumoniaMNIST(split="val", transform=eval_tf, download=True)
test_ds  = PneumoniaMNIST(split="test", transform=eval_tf, download=True)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
for p in model.parameters(): p.requires_grad = False
model.fc = nn.Sequential(nn.Dropout(0.30), nn.Linear(model.fc.in_features, 2))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)

def epoch(loader, training=False):
    model.train(training)
    losses, ytrue, ypred = [], [], []
    for x,y in loader:
        x=x.to(DEVICE); y=y.squeeze().long().to(DEVICE)
        if training: optimizer.zero_grad()
        logits=model(x); loss=criterion(logits,y)
        if training:
            loss.backward(); optimizer.step()
        losses.append(loss.item())
        ytrue += y.detach().cpu().tolist()
        ypred += logits.argmax(1).detach().cpu().tolist()
    return np.mean(losses), accuracy_score(ytrue, ypred)

history=[]; best=0
for e in range(5):
    tl,ta=epoch(train_loader,True)
    vl,va=epoch(val_loader,False)
    history.append([e+1,tl,ta,vl,va])
    print(f"Epoch {e+1}: train_acc={ta:.4f}, val_acc={va:.4f}")
    if va > best:
        best=va
        torch.save(model.state_dict(), OUT/"mediscan_resnet50.pt")

pd.DataFrame(history,columns=["epoch","train_loss","train_acc","val_loss","val_acc"]).to_csv(OUT/"training_history.csv",index=False)

model.load_state_dict(torch.load(OUT/"mediscan_resnet50.pt",map_location=DEVICE))
model.eval()
ys=[]; preds=[]; probs=[]
with torch.no_grad():
    for x,y in test_loader:
        logits=model(x.to(DEVICE))
        p=torch.softmax(logits,1)[:,1]
        ys += y.squeeze().tolist()
        probs += p.cpu().tolist()
        preds += (p>=0.5).long().cpu().tolist()

acc=accuracy_score(ys,preds)
precision,recall,f1,_=precision_recall_fscore_support(ys,preds,average="binary",zero_division=0)
auc=roc_auc_score(ys,probs)
metrics={"accuracy":acc,"precision":precision,"recall":recall,"f1":f1,"roc_auc":auc}
json.dump(metrics,open(OUT/"metrics.json","w"),indent=2)
print(metrics)

cm=confusion_matrix(ys,preds)
plt.figure(figsize=(5,4))
sns.heatmap(cm,annot=True,fmt="d",xticklabels=["Normal","Pneumonia"],yticklabels=["Normal","Pneumonia"])
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix"); plt.tight_layout()
plt.savefig(OUT/"confusion_matrix.png",dpi=180); plt.show()

fpr,tpr,_=roc_curve(ys,probs)
plt.figure(figsize=(5,4)); plt.plot(fpr,tpr,label=f"AUC={auc:.3f}"); plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve"); plt.legend(); plt.tight_layout()
plt.savefig(OUT/"roc_curve.png",dpi=180); plt.show()

h=pd.DataFrame(history,columns=["epoch","train_loss","train_acc","val_loss","val_acc"])
plt.figure(figsize=(6,4)); plt.plot(h.epoch,h.train_acc,label="Train"); plt.plot(h.epoch,h.val_acc,label="Validation")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Training/Validation Accuracy"); plt.legend(); plt.tight_layout()
plt.savefig(OUT/"accuracy_curve.png",dpi=180); plt.show()

plt.figure(figsize=(6,4)); plt.plot(h.epoch,h.train_loss,label="Train"); plt.plot(h.epoch,h.val_loss,label="Validation")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training/Validation Loss"); plt.legend(); plt.tight_layout()
plt.savefig(OUT/"loss_curve.png",dpi=180); plt.show()

# Grad-CAM: 12 examples
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

cam=GradCAM(model=model,target_layers=[model.layer4[-1]])
grad_dir=OUT/"gradcam"; grad_dir.mkdir(exist_ok=True)
mean=np.array([0.485,0.456,0.406]); std=np.array([0.229,0.224,0.225])

for i in range(12):
    x,y=test_ds[i]
    xb=x.unsqueeze(0).to(DEVICE)
    with torch.no_grad(): pred=int(model(xb).argmax(1).item())
    gray=cam(input_tensor=xb,targets=[ClassifierOutputTarget(pred)])[0]
    rgb=np.clip(x.permute(1,2,0).numpy()*std+mean,0,1)
    overlay=show_cam_on_image(rgb,gray,use_rgb=True)
    plt.imsave(grad_dir/f"gradcam_{i+1:02d}.png",overlay)

# Copy trained model for Streamlit
import shutil
shutil.copy(OUT/"mediscan_resnet50.pt", "mediscan_app_model.pt")
print("DONE. Copy mediscan_app_model.pt to app/ and rename it to mediscan_resnet50.pt.")
