
# tasks/module1.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from modules.module1_loss import alpha_diversity_loss


class Module1Trainer:
    """
    Trainer cho NewsFactorizationModule.

    Sử dụng:
        trainer = Module1Trainer(model, optimizer, scheduler, cfg, device)
        train_loss, train_acc = trainer.train_epoch(train_loader)
        val_loss, val_acc, val_prec, val_rec, val_f1 = trainer.evaluate(val_loader)
        trainer.export_features(loader, output_path)
        trainer.save_checkpoint(path, epoch, best_val_loss)
        start_epoch, best_val_loss = trainer.load_checkpoint(path)
    """

    def __init__(self, model, optimizer, scheduler, cfg, device):
        self.model     = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg       = cfg
        self.device    = device

        train_cfg = cfg.training
        self.lambda_div     = train_cfg.lambda_div
        self.lambda_entropy = getattr(train_cfg, 'lambda_entropy', 0.01)
        self.rdrop_alpha    = getattr(train_cfg.rdrop, 'alpha', 0.5)
        self.grad_clip      = train_cfg.grad_clip

        # FIX: model._encode trả về log(probs) làm "logits"
        # → dùng NLLLoss thay vì CrossEntropyLoss để tránh double-softmax
        self.criterion_train = nn.NLLLoss(
            label_smoothing=model.label_smoothing
        )
        self.criterion_eval = nn.NLLLoss()

    # ═══════════════════════════════════════════════════════════════════════
    # TRAIN
    # ═══════════════════════════════════════════════════════════════════════

    def train_epoch(self, loader) -> tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(loader, desc="Training", leave=False)
        for batch in pbar:
            ids, masks, srl, counts, factors, labels = self._unpack(batch)

            self.optimizer.zero_grad()

            # R-Drop: 2 forward passes với dropout khác nhau
            logits1, probs1 = self.model(ids, masks, srl, counts, factors)
            logits2, probs2 = self.model(ids, masks, srl, counts, factors)

            # Task loss — logits đã là log(probs) nên dùng NLLLoss
            task_loss = (
                self.criterion_train(logits1, labels) +
                self.criterion_train(logits2, labels)
            ) / 2

            # KL divergence (R-Drop) — tính trực tiếp trên log_probs
            kl_loss = (
                F.kl_div(logits1, probs2, reduction='batchmean') +
                F.kl_div(logits2, probs1, reduction='batchmean')
            ) / 2

            # Diversity regularization trên W_alpha gates
            div_loss = alpha_diversity_loss(
                self.model,
                lambda_div     = self.lambda_div,
                lambda_entropy = self.lambda_entropy,
            )

            loss = task_loss + self.rdrop_alpha * kl_loss + div_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip
            )
            self.optimizer.step()

            total_loss += task_loss.item() * len(labels)
            correct    += (logits1.argmax(-1) == labels).sum().item()
            total      += len(labels)
            pbar.set_postfix(
                loss=f"{total_loss/total:.4f}",
                acc =f"{correct/total:.4f}",
            )

        return total_loss / total, correct / total

    # ═══════════════════════════════════════════════════════════════════════
    # EVALUATE
    # ═══════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def evaluate(self, loader) -> tuple[float, float, float, float, float]:
        self.model.eval()
        total_loss, total = 0.0, 0
        all_preds, all_labels = [], []

        pbar = tqdm(loader, desc="Evaluating", leave=False)
        for batch in pbar:
            ids, masks, srl, counts, factors, labels = self._unpack(batch)

            logits, _ = self.model(ids, masks, srl, counts, factors)
            loss  = self.criterion_eval(logits, labels)
            preds = logits.argmax(dim=-1)

            total_loss += loss.item() * len(labels)
            total      += len(labels)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        acc       = (np.array(all_preds) == np.array(all_labels)).mean()
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall    = recall_score   (all_labels, all_preds, average='macro', zero_division=0)
        f1        = f1_score       (all_labels, all_preds, average='macro', zero_division=0)

        return total_loss / total, acc, precision, recall, f1

    # ═══════════════════════════════════════════════════════════════════════
    # EXPORT FEATURES
    # ═══════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def export_features(self, loader, output_path: str = "extracted_features.csv"):
        """
        Xuất SRL/SDPG features + dự đoán ra CSV.
        Mỗi row = 1 (sample × news) pair.
        Để lấy 1 row/sample: df.drop_duplicates(['CODE', 'trade_date'])
        """
        self.model.eval()
        all_records = []

        print(f"Extracting features + predictions → {output_path}")
        for batch in tqdm(loader, desc="Exporting"):
            ids     = batch['input_ids'].to(self.device)
            masks   = batch['attention_mask'].to(self.device)
            srl     = batch['srl_spans']
            counts  = batch['news_counts'].to(self.device)
            factors = batch['stock_factors'].to(self.device)
            labels  = batch['label']   # CPU

            B   = ids.size(0)
            enc = self.model._encode(ids, masks, srl, counts, factors)

            probs_cpu   = enc['probs'].cpu().tolist()
            pred_labels = enc['logits'].argmax(-1).cpu().tolist()

            for i in range(B):
                real_n = counts[i].item()
                p      = probs_cpu[i]
                for j in range(real_n):
                    all_records.append({
                        'CODE':          batch['code'][i],
                        'trade_date':    batch['trade_date'][i],
                        'news_idx':      j,
                        'label':         labels[i].item(),
                        'pred_label':    pred_labels[i],
                        'pred_prob_neg': round(p[0], 6),
                        'pred_prob_neu': round(p[1], 6),
                        'pred_prob_pos': round(p[2], 6),
                        'eV':            enc['eV'][i, j].cpu().tolist(),
                        'eA0':           enc['eA0'][i, j].cpu().tolist(),
                        'eA1':           enc['eA1'][i, j].cpu().tolist(),
                        'G_VA0':         enc['G_VA0'][i, j].cpu().tolist(),
                        'G_VA1':         enc['G_VA1'][i, j].cpu().tolist(),
                        'G_A0A1':        enc['G_A0A1'][i, j].cpu().tolist(),
                        'e_SDPG':        enc['e_SDPG'][i, j].cpu().tolist(),
                    })

        pd.DataFrame(all_records).to_csv(output_path, index=False)
        print(f"✅ {len(all_records)} rows saved → {output_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # CHECKPOINT
    # ═══════════════════════════════════════════════════════════════════════

    def save_checkpoint(self, path: str, epoch: int, best_val_loss: float):
        torch.save({
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch':                epoch,
            'best_val_loss':        best_val_loss,
        }, path)

    def load_checkpoint(self, path: str) -> tuple[int, float]:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        epoch         = ckpt.get('epoch', 0)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"✅ Loaded checkpoint từ epoch {epoch} | best_val_loss={best_val_loss:.4f}")
        return epoch, best_val_loss

    # ═══════════════════════════════════════════════════════════════════════
    # HELPER
    # ═══════════════════════════════════════════════════════════════════════

    def _unpack(self, batch):
        return (
            batch['input_ids'].to(self.device),
            batch['attention_mask'].to(self.device),
            batch['srl_spans'],                      # list — không to(device)
            batch['news_counts'].to(self.device),
            batch['stock_factors'].to(self.device),
            batch['label'].to(self.device),
        )