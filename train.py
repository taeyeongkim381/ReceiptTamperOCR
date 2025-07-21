import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import RobertaTokenizer

from model import get_model
from data import ReceiptForgeryDataset
from trainer import ImageTrainer, TextTrainer
from util import set_seed, FocalLoss

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # ──────────────────────────────
    # 기본 세팅
    # ──────────────────────────────
    logger = logging.getLogger(__name__)
    logger.info("\n" + OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    # Hydra가 생성한 현재 작업 디렉토리
    work_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.info(f"💾 Outputs will be saved in: {work_dir}")

    # ──────────────────────────────
    # 데이터셋 & 로더 준비
    # ──────────────────────────────
    base_dir = cfg.base_dir
    img_transform = transforms.Compose([
        transforms.Resize((512, 512)),

        # 색상 변형 (너무 강하면 안됨)
        transforms.ColorJitter(
            brightness=0.1,   # 밝기 변화 줄임
            contrast=0.1,     # 대비 변화 줄임
            saturation=0.05,  # 채도 변화 줄임
            hue=0.01          # 색조 변화 줄임
        ),

        # 랜덤 회전 (영수증은 보통 약간만 기울어짐)
        transforms.RandomRotation(degrees=1.5, fill=255),

        # 약한 affine 변형 (이동/스케일 범위 축소)
        transforms.RandomAffine(
            degrees=0,
            translate=(0.02, 0.02),  # 2% 범위 이동
            shear=1,                 # 기울기 줄임
            scale=(0.98, 1.02),      # 스케일 변화를 최소화
            fill=255
        ),

        # GaussianBlur 확률도 줄이고 kernel 작게
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    train_dataset = ReceiptForgeryDataset(
        csv_path=os.path.join(base_dir, cfg.train_csv),
        base_dir=base_dir,
        split="train",
        image_transform=img_transform,
        text_tokenizer=tokenizer,
        max_length=cfg.max_length
    )
    val_dataset = ReceiptForgeryDataset(
        csv_path=os.path.join(base_dir, cfg.val_csv),
        base_dir=base_dir,
        split="val",
        image_transform=img_transform,
        text_tokenizer=tokenizer,
        max_length=cfg.max_length
    )
    test_dataset = ReceiptForgeryDataset(
        csv_path=os.path.join(base_dir, cfg.test_csv),
        base_dir=base_dir,
        split="test",
        image_transform=img_transform,
        text_tokenizer=tokenizer,
        max_length=cfg.max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    # ──────────────────────────────
    # 모델 선택 및 Trainer 준비
    # ──────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.model_type == "image":
        model = get_model("resnet50")
        optimizer = optim.Adam(model.parameters(), lr=cfg.image_lr)
        # ==========================
        # ✨ FocalLoss 적용
        # ==========================
        criterion = FocalLoss(gamma=2.0, alpha=[1.0, 2.0], reduction='mean')
       
        trainer = ImageTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            logger=logger,
            scheduler=None,
            device=device,
            log_dir=os.path.join(work_dir, "runs"),
            patience=cfg.patience,
            save_path=os.path.join(work_dir, "best_image_model.pth")
        )
    elif cfg.model_type == "text":
        model = get_model("roberta")
        optimizer = optim.AdamW(model.parameters(), lr=cfg.text_lr)
        # ==========================
        # ✨ FocalLoss 적용
        # ==========================
        criterion = FocalLoss(gamma=2.0, alpha=[1.0, 2.0], reduction='mean')
        trainer = TextTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            logger=logger,
            scheduler=None,
            device=device,
            log_dir=os.path.join(work_dir, "runs"),
            patience=cfg.patience,
            save_path=os.path.join(work_dir, "best_text_model.pth")
        )
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type}")

    # ──────────────────────────────
    # 학습 시작
    # ──────────────────────────────
    trainer.training(train_loader, val_loader, cfg.epochs)
    trainer.test(test_loader)

if __name__ == "__main__":
    main()
