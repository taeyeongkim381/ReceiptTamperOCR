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
from util import set_seed

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
        # 크기 조정
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        
        # 밝기/대비/채도/색조 약간 변화
        transforms.ColorJitter(
            brightness=0.2,  # 밝기
            contrast=0.2,    # 대비
            saturation=0.1,  # 채도
            hue=0.02         # 색조
        ),

        # 좌우 뒤집기 (영수증은 대칭적이라 큰 문제 없지만 세로텍스트가 있다면 False)
        # transforms.RandomHorizontalFlip(p=0.5),  # 필요하면 사용

        # 약간의 회전 (영수증이 약간 기울어질 수 있음)
        transforms.RandomRotation(degrees=2.0, expand=False, fill=255),

        # 랜덤한 미세한 affine 변형 (약간의 이동/기울임)
        transforms.RandomAffine(
            degrees=0,
            translate=(0.02, 0.02),  # 최대 2% 이동
            shear=2,                # 약간의 기울임
            fill=255                # 배경색을 흰색으로 채움
        ),

        # 텐서 변환
        transforms.ToTensor(),

        # Normalize (필요하다면 ImageNet 기준)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
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
        criterion = nn.CrossEntropyLoss()
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
        criterion = nn.CrossEntropyLoss()
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
