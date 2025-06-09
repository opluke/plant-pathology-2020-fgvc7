"""
Plant Pathology 2020 – FGVC7
完整訓練 / 推論腳本
"""

import argparse, os, random
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.transforms.autoaugment import AutoAugment, AutoAugmentPolicy
from sklearn.model_selection import KFold
from tqdm import tqdm

# ========= 1. 損失函數 =========
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, inputs, targets):
        bce = self.bce(inputs, targets)
        pt  = torch.exp(-bce)
        loss = self.alpha * (1-pt) ** self.gamma * bce
        return loss.mean() if self.reduction == 'mean' else loss.sum()

# ========= 2. Dataset =========
class PlantDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, transform=None):
        self.df, self.img_dir, self.transform = df.reset_index(drop=True), img_dir, transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(os.path.join(self.img_dir, f"{row.image_id}.jpg")).convert("RGB")
        label = row[["healthy", "multiple_diseases", "rust", "scab"]].values.astype(np.float32)
        if self.transform: img = self.transform(img)
        return img, torch.tensor(label)

class TTADataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str):
        self.df, self.img_dir = df.reset_index(drop=True), img_dir

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, f"{row.image_id}.jpg")).convert("RGB")
        return img, row.image_id

# ========= 3. Model Factory =========
def get_model(name: str) -> nn.Module:
    if name == "efficientnet_b3":
        m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier[1].in_features, 4))
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, 4)
    elif name == "densenet201":
        m = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        m.classifier = nn.Linear(m.classifier.in_features, 4)
    elif name == "convnext_base":
        m = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, 4)
    else:
        raise ValueError(f"Unknown model {name}")
    return m

# ========= 4. 訓練流程 =========
def train_model(model, train_loader, val_loader, device, lr=1e-4, epochs=5, amp=True):
    model.to(device)
    optim_ = optim.Adam(model.parameters(), lr=lr)
    criterion = FocalLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best_f1 = -1

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optim_.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp):
                out = torch.sigmoid(model(x))
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optim_)
            scaler.update()
            pbar.set_postfix(loss=loss.item())

        # -- 驗證 --
        model.eval()
        t_correct = t_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out  = torch.sigmoid(model(x))
                pred, true = out.argmax(1), y.argmax(1)
                t_correct += (pred == true).sum().item()
                t_total   += y.size(0)
        acc = t_correct / t_total
        if acc > best_f1:
            best_f1 = acc
            torch.save(model.state_dict(), f"{model.name}_best.pth" if hasattr(model,"name") else "best.pth")
        print(f"  ‣ Val Acc = {acc:.4f} (best {best_f1:.4f})")

# ========= 5. TTA 子集 =========
class TTASubset(Dataset):
    def __init__(self, dataset, tta_transform):
        self.dataset, self.tt = dataset, tta_transform
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        img, img_id = self.dataset[idx]
        return self.tt(img), img_id

# ========= 6. Ensemble 推論 =========
def predict_ensemble(models, dataset, transforms_list, device):
    for m in models:
        m.to(device)
        m.eval()

    n_samples = len(dataset)
    ids       = [dataset[i][1] for i in range(n_samples)]
    all_preds = [[] for _ in range(n_samples)]

    with torch.no_grad():
        for tta in transforms_list:
            loader = DataLoader(
                TTASubset(dataset, tta),
                batch_size=32,
                shuffle=False,
                drop_last=False
            )

            global_ptr = 0             # ← 逐張累計，不再依賴 batch_idx
            for x, _ in loader:
                x = x.to(device)
                outs  = torch.stack([torch.sigmoid(m(x)).cpu()
                                     for m in models])   # [n_models, bs, 4]
                preds = outs.mean(0)                     # [bs, 4]
                bs    = preds.size(0)

                for j in range(bs):
                    all_preds[global_ptr].append(preds[j])
                    global_ptr += 1

            assert global_ptr == n_samples, \
                "資料數與累計樣本數不一致，請檢查 DataLoader。"

    # 生成結果
    results = [torch.stack(p).mean(0).numpy() for p in all_preds]
    return ids, results

# ========= 7. 主程式 =========
def main(args):
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    df, test_df = pd.read_csv("train.csv"), pd.read_csv("test.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_tf = transforms.Compose([
        transforms.Resize((224,224)),
        AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # ---------- 訓練 ----------
    if not args.inference:
        folds = range(5) if args.train_all else [args.fold]
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(df)):
            if fold not in folds: continue
            print(f"\n===== Fold {fold} =====")
            train_ds = PlantDataset(df.iloc[tr_idx], "images", base_tf)
            val_ds   = PlantDataset(df.iloc[va_idx], "images", base_tf)
            tl = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=False)
            vl = DataLoader(val_ds,   batch_size=32, shuffle=False)

            for name in args.models:
                print(f"\n--- Training {name} ---")
                model = get_model(name)
                model.name = f"{name}_fold{fold}"  # type: ignore[attr-defined]
                train_model(model, tl, vl, device,
                            lr=1e-4, epochs=5, amp=args.amp)

    # ---------- 推論 ----------
    print("\n===== Inference =====")
    test_set = TTADataset(test_df, "images")
    tta_list = [
        base_tf,
        transforms.Compose([transforms.Resize((224,224)),
                            transforms.RandomHorizontalFlip(p=1.0),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        transforms.Compose([transforms.Resize((224,224)),
                            transforms.RandomRotation(15),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    ]

    models_all = []
    folds_to_use = range(5) if args.train_all else [args.fold]
    for fold in folds_to_use:
        for name in args.models:
            m = get_model(name)
            m.load_state_dict(torch.load(f"{name}_fold{fold}.pth", map_location=device, weights_only=True))
            models_all.append(m)

    ids, preds = predict_ensemble(models_all, test_set, tta_list, device)
    out_df = pd.DataFrame(preds, columns=["healthy","multiple_diseases","rust","scab"])
    out_df.insert(0,"image_id",ids)
    out_df.to_csv("submission_ensemble_upgraded.csv", index=False)
    print("✅ submission_ensemble_upgraded.csv saved!")

# ========= CLI =========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0, help="只訓練 / 推論第 n 折 (0-4)")
    parser.add_argument("--train_all", action="store_true", help="訓練並推論所有 5 折")
    parser.add_argument("--inference", action="store_true", help="僅進行推論，不訓練")
    parser.add_argument("--amp", action="store_true", help="啟用混合精度")
    parser.add_argument("--models", nargs="+", default=["efficientnet_b3","resnet50","densenet201","convnext_base"])
    main(parser.parse_args())
