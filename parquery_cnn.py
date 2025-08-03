import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import multiprocessing
from torchvision.models import ResNet18_Weights
from PIL import ImageOps

def train():
  # 1) 데이터 디렉토리 & 하이퍼파라미터
  data_dir      = 'D:/Projects/vision/dataset6'    # train/car, train/no_car, val/... 구조
  batch_size    = 32
  num_epochs    = 10
  learning_rate = 1e-4
  device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # 2) 전처리: (H=30, W=100) 원본 크기 그대로 사용
  transform = transforms.Compose([
      transforms.ToTensor(),
      # transforms.Normalize([0.5], [0.5])  
      transforms.Normalize([0.485,0.456,0.406],
                          [0.229,0.224,0.225])
  ])

  # 2) 전처리 (Augmentation 포함)
  train_transform = transforms.Compose([
      transforms.Resize((30, 100)),  # 원본 크기 고정
      # 밝기·대비 랜덤 변화
      transforms.RandomApply(
          [transforms.ColorJitter(brightness=0.7, contrast=0.7)],
          p=0.8
      ),
      # 히스토그램 평활화
      transforms.RandomApply(
          [transforms.Lambda(lambda img: ImageOps.equalize(img))],
          p=0.5
      ),
      transforms.RandomGrayscale(p=0.2),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(
          [0.485, 0.456, 0.406],     # ImageNet 평균
          [0.229, 0.224, 0.225]      # ImageNet 표준편차
      ),
  ])
  val_transform = transforms.Compose([
      transforms.Resize((30, 100)),
      transforms.ToTensor(),
      transforms.Normalize(
          [0.485, 0.456, 0.406],
          [0.229, 0.224, 0.225]
      ),
  ])

  # 3) 데이터셋 & 로더
  train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
  val_ds   = datasets.ImageFolder(os.path.join(data_dir, 'val'),   transform=val_transform)

  num_samples = len(train_ds)
  num_classes = len(train_ds.classes)
  # 클래스별 샘플 수 집계
  targets = torch.tensor(train_ds.targets)
  class_counts = torch.bincount(targets)
  # weight_i = N / (K * count_i)
  class_weights = (num_samples / (num_classes * class_counts.float())).to(device)

  train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
  val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

  # 4) 모델 정의 (2‑class 분류)
  model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
  model.fc = nn.Linear(model.fc.in_features, num_classes)
  model = model.to(device)

  # 5) 손실함수 + 옵티마이저
  criterion = nn.CrossEntropyLoss(weight=class_weights)
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)

  best_acc = 0.0

  # 6) 학습 루프
  for epoch in range(1, num_epochs+1):
      # --- Training ---
      model.train()
      running_loss = 0.0
      running_corrects = 0
      for inputs, labels in train_loader:
          inputs = inputs.to(device)
          labels = labels.to(device)

          optimizer.zero_grad()
          outputs = model(inputs)                   # (B,2) logits
          loss = criterion(outputs, labels)         
          loss.backward()
          optimizer.step()

          running_loss += loss.item() * inputs.size(0)
          preds = torch.argmax(outputs, dim=1)
          running_corrects += (preds == labels).sum().item()

      epoch_loss = running_loss / num_samples
      epoch_acc  = running_corrects / num_samples

      # --- Validation ---
      # --- Validation Phase ---
      model.eval()
      val_loss = 0.0
      val_corrects = 0
      val_samples = len(val_ds)

      # ROC용 리스트 초기화
      y_true = []
      y_scores = []

      with torch.no_grad():
          for inputs, labels in val_loader:
              inputs = inputs.to(device)
              labels = labels.to(device)

              outputs = model(inputs)               # (B,2) 로짓
              loss = criterion(outputs, labels)

              # loss & accuracy 집계
              val_loss += loss.item() * inputs.size(0)
              preds = torch.argmax(outputs, dim=1)
              val_corrects += (preds == labels).sum().item()

              # ROC용 데이터 수집
              probs = torch.softmax(outputs, dim=1) # (B,2) 확률
              y_true.extend(labels.cpu().tolist())
              y_scores.extend(probs[:,1].cpu().tolist())  # 클래스 'car' 확률

      val_epoch_loss = val_loss / val_samples
      val_epoch_acc  = val_corrects / val_samples

      # --- ROC 커브 그리기 (예: 매 에폭마다) ---
      # from sklearn.metrics import roc_curve, auc
      # import matplotlib.pyplot as plt

      # fpr, tpr, _ = roc_curve(y_true, y_scores)
      # roc_auc = auc(fpr, tpr)

      # plt.figure(figsize=(5,5))
      # plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
      # plt.plot([0,1], [0,1], '--', color='gray')
      # plt.xlabel('False Positive Rate')
      # plt.ylabel('True Positive Rate')
      # plt.title(f'Epoch {epoch} ROC Curve')
      # plt.legend(loc='lower right')
      # plt.show()  # 또는 plt.savefig(f'roc_epoch{epoch}.png')
      # --- ROC 커브 그리기 (예: 매 에폭마다) ---

      # --- 베스트 모델 저장 ---
      if val_epoch_acc >= best_acc:
          best_acc = val_epoch_acc
          torch.save(model.state_dict(), "best_resnet18_07_black.pth")
          print(f"[Saved] epoch {epoch}, val_acc={best_acc:.4f}")

      # --- 로그 출력 ---
      print(
          f"Epoch {epoch}/{num_epochs}  "
          f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}  "
          f"Val   Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}"
      )

if __name__ == "__main__":
    # Windows spawn 안전화
    multiprocessing.freeze_support()
    train()