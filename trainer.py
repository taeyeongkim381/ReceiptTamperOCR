import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, classification_report
import numpy as np
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple, Dict, Any


class ImageTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        logger: Any,
        scheduler: Optional[Any] = None,
        device: str = "cuda",
        log_dir: str = "./runs_image",
        patience: int = 3,
        save_path: str = "./best_image_model.pt",
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.logger = logger
        self.scheduler = scheduler
        self.device = device
        self.writer = SummaryWriter(log_dir=log_dir)
        self.patience = patience
        self.best_auc = 0.0
        self.no_improve_count = 0
        self.save_path = save_path

    def train(self, loader: torch.utils.data.DataLoader, epoch: int) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        all_labels, all_preds = [], []

        pbar = tqdm(loader, desc=f"[Image Train {epoch}]")
        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].long().to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(probs.detach().cpu().numpy())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if self.scheduler:
            self.scheduler.step()

        avg_loss = running_loss / len(loader)
        auc = roc_auc_score(all_labels, all_preds)
        self.logger.info(f"[Image Train] Epoch {epoch} Loss {avg_loss:.4f} AUC {auc:.4f}")
        return avg_loss, auc

    def val(self, loader: torch.utils.data.DataLoader, epoch: int) -> Tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        all_labels, all_preds = [], []

        with torch.no_grad():
            pbar = tqdm(loader, desc=f"[Image Val {epoch}]")
            for batch in pbar:
                images = batch["image"].to(self.device)
                labels = batch["label"].long().to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, labels)
                running_loss += loss.item()

                probs = torch.softmax(logits, dim=1)[:, 1]
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(probs.cpu().numpy())

        avg_loss = running_loss / len(loader)
        auc = roc_auc_score(all_labels, all_preds)
        self.writer.add_scalar("Loss/val", avg_loss, epoch)
        self.writer.add_scalar("AUC/val", auc, epoch)
        self.logger.info(f"[Image Val] Epoch {epoch} Loss {avg_loss:.4f} AUC {auc:.4f}")
        self.logger.info("\n" + classification_report(
            np.array(all_labels),
            (np.array(all_preds) >= 0.5).astype(int),
            digits=4
        ))

        if auc > self.best_auc:
            self.best_auc = auc
            self.no_improve_count = 0
            torch.save(self.model.state_dict(), self.save_path)
            self.logger.info(f"Best model saved with AUC {auc:.4f} -> {self.save_path}")
        else:
            self.no_improve_count += 1
            self.logger.info(f"EarlyStopping counter: {self.no_improve_count}/{self.patience}")

        return avg_loss, auc

    def test(self, loader: torch.utils.data.DataLoader) -> float:
        self.model.eval()
        all_labels, all_preds = [], []

        with torch.no_grad():
            pbar = tqdm(loader, desc="[Image Test]")
            for batch in pbar:
                images = batch["image"].to(self.device)
                labels = batch["label"].long().to(self.device)

                logits = self.model(images)
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(probs.cpu().numpy())

        auc = roc_auc_score(all_labels, all_preds)
        self.logger.info(f"[Image Test] AUC {auc:.4f}")
        self.logger.info("\n" + classification_report(
            np.array(all_labels),
            (np.array(all_preds) >= 0.5).astype(int),
            digits=4
        ))
        return auc

    def training(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, epochs: int) -> None:
        for epoch in range(1, epochs + 1):
            train_loss, train_auc = self.train(train_loader, epoch)
            val_loss, val_auc = self.val(val_loader, epoch)

            self.writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
            self.writer.add_scalars("AUC", {"train": train_auc, "val": val_auc}, epoch)

            if self.no_improve_count >= self.patience:
                self.logger.info("Early stopping triggered.")
                break


class TextTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        logger: Any,
        scheduler: Optional[Any] = None,
        device: str = "cuda",
        log_dir: str = "./runs_text",
        patience: int = 3,
        save_path: str = "./best_text_model.pt",
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.logger = logger
        self.scheduler = scheduler
        self.device = device
        self.writer = SummaryWriter(log_dir=log_dir)
        self.patience = patience
        self.best_auc = 0.0
        self.no_improve_count = 0
        self.save_path = save_path

    def train(self, loader: torch.utils.data.DataLoader, epoch: int) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        all_labels, all_preds = [], []

        pbar = tqdm(loader, desc=f"[Text Train {epoch}]")
        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].long().to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(probs.detach().cpu().numpy())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if self.scheduler:
            self.scheduler.step()

        avg_loss = running_loss / len(loader)
        auc = roc_auc_score(all_labels, all_preds)
        self.writer.add_scalar("Loss/train", avg_loss, epoch)
        self.writer.add_scalar("AUC/train", auc, epoch)
        self.logger.info(f"[Text Train] Epoch {epoch} Loss {avg_loss:.4f} AUC {auc:.4f}")
        return avg_loss, auc

    def val(self, loader: torch.utils.data.DataLoader, epoch: int) -> Tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        all_labels, all_preds = [], []

        with torch.no_grad():
            pbar = tqdm(loader, desc=f"[Text Val {epoch}]")
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].long().to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = self.criterion(logits, labels)
                running_loss += loss.item()

                probs = torch.softmax(logits, dim=1)[:, 1]
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(probs.cpu().numpy())

        avg_loss = running_loss / len(loader)
        auc = roc_auc_score(all_labels, all_preds)
        self.logger.info(f"[Text Val] Epoch {epoch} Loss {avg_loss:.4f} AUC {auc:.4f}")
        self.logger.info("\n" + classification_report(
            np.array(all_labels),
            (np.array(all_preds) >= 0.5).astype(int),
            digits=4
        ))

        if auc > self.best_auc:
            self.best_auc = auc
            self.no_improve_count = 0
            torch.save(self.model.state_dict(), self.save_path)
            self.logger.info(f"Best model saved with AUC {auc:.4f} -> {self.save_path}")
        else:
            self.no_improve_count += 1
            self.logger.info(f"EarlyStopping counter: {self.no_improve_count}/{self.patience}")

        return avg_loss, auc

    def test(self, loader: torch.utils.data.DataLoader) -> float:
        self.model.eval()
        all_labels, all_preds = [], []

        with torch.no_grad():
            pbar = tqdm(loader, desc="[Text Test]")
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].long().to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(probs.cpu().numpy())

        auc = roc_auc_score(all_labels, all_preds)
        self.logger.info(f"[Text Test] AUC {auc:.4f}")
        self.logger.info("\n" + classification_report(
            np.array(all_labels),
            (np.array(all_preds) >= 0.5).astype(int),
            digits=4
        ))
        return auc

    def training(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, epochs: int) -> None:
        for epoch in range(1, epochs + 1):
            train_loss, train_auc = self.train(train_loader, epoch)
            val_loss, val_auc = self.val(val_loader, epoch)

            self.writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
            self.writer.add_scalars("AUC", {"train": train_auc, "val": val_auc}, epoch)

            if self.no_improve_count >= self.patience:
                self.logger.info("Early stopping triggered.")
                break
