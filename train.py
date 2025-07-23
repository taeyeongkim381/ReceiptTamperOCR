import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import RobertaTokenizerFast

from data import TextBinaryDataset, ImageBinaryDataset, split_train_valid
from model import get_model
from trainer import ImageTrainer, TextTrainer
from util import set_seed


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main training entry point with Hydra configuration."""
    logger = logging.getLogger(__name__)
    logger.info("\n" + OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    work_dir: str = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.info(f"Outputs will be saved in: {work_dir}")

    train_csv_full: str = os.path.join(cfg.base_dir, cfg.train_csv)
    test_csv_full: str = os.path.join(cfg.base_dir, cfg.test_csv)
    train_split, valid_split = split_train_valid(train_csv_full)

    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        transforms.RandomRotation(degrees=2.0, fill=255),
        transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), shear=2, fill=255),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg.model_type == "image":
        train_ds = ImageBinaryDataset(train_split, transform=img_transform)
        valid_ds = ImageBinaryDataset(valid_split, transform=img_transform)
        test_ds = ImageBinaryDataset(test_csv_full, transform=img_transform)

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
        val_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

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
        train_ds = TextBinaryDataset(train_split, tokenizer, max_len=cfg.max_length)
        valid_ds = TextBinaryDataset(valid_split, tokenizer, max_len=cfg.max_length)
        test_ds = TextBinaryDataset(test_csv_full, tokenizer, max_len=cfg.max_length)

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
        val_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

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

    trainer.training(train_loader, val_loader, cfg.epochs)
    trainer.test(test_loader)


if __name__ == "__main__":
    main()
