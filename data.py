import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
from torchvision import transforms
from transformers import RobertaTokenizerFast

# ============================================================
# 1. 이미지용 Dataset
# ============================================================
class ImageBinaryDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        df = pd.read_csv(csv_path)
        self.image_paths = df["image_path"].tolist()
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = int(self.labels[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long)
        }


# ============================================================
# 2. 텍스트용 Dataset (RoBERTa 토크나이저)
# ============================================================
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

class TextBinaryDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=512):
        df = pd.read_csv(csv_path)
        self.txt_paths = df["txt_path"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.txt_paths)

    def __getitem__(self, idx):
        txt_path = self.txt_paths[idx]
        label = int(self.labels[idx])
        # txt 파일 읽기
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


# ============================================================
# 3. Dataloader 생성 함수
# ============================================================
def split_train_valid(csv_path, valid_ratio=0.2):
    df = pd.read_csv(csv_path)
    train_df, valid_df = train_test_split(df, test_size=valid_ratio, stratify=df["label"], random_state=42)
    base_dir = os.path.dirname(csv_path)
    train_split = os.path.join(base_dir, "_train_split.csv")
    valid_split = os.path.join(base_dir, "_valid_split.csv")
    train_df.to_csv(train_split, index=False)
    valid_df.to_csv(valid_split, index=False)
    return train_split, valid_split
