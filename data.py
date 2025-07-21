import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch

class ReceiptForgeryDataset(Dataset):
    def __init__(self, csv_path:str, base_dir:str, split:str,
                 image_transform=None, text_tokenizer=None, max_length:int=256):
        """
        csv_path: train.csv / val.csv / test.csv 경로
        base_dir: "/workspace/data/findit2"
        split: "train" | "val" | "test"
        image_transform: torchvision.transforms.Compose
        text_tokenizer: HuggingFace tokenizer
        max_length: 토큰 최대 길이
        """
        self.df = pd.read_csv(csv_path)
        self.base_dir = base_dir
        self.split = split
        self.image_transform = image_transform
        self.text_tokenizer = text_tokenizer
        self.max_length = max_length

        # 이미지와 OCR 파일이 모두 들어있는 폴더
        self.data_dir = os.path.join(base_dir, split)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- 이미지 로드 ---
        image_name = row['image']   # 예: "X000001.png"
        image_path = os.path.join(self.data_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)
        else:
            from torchvision import transforms
            image = transforms.ToTensor()(image)

        # --- OCR 로드 ---
        ocr_name = row['ocr']       # 예: "X000001.txt"
        ocr_path = os.path.join(self.data_dir, ocr_name)
        if os.path.exists(ocr_path):
            with open(ocr_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        else:
            text = ""

        # --- 토크나이징 ---
        if self.text_tokenizer:
            enc = self.text_tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
        else:
            input_ids = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_length, dtype=torch.long)

        # --- 라벨 ---
        label = int(row['forged'])

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long)
        }
    
