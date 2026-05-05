# # main.py
# import os
# import numpy as np
# import torch
# from omegaconf import OmegaConf
# from transformers import AutoTokenizer
# import argparse

# from builders.registry import register_model
# from builders.model_builder import build_model
# from data_utils.module1 import (
#     load_data,
#     group_by_stock_date,
#     make_sampler,
#     NewsFactorDataset,
#     collate_fn,
# )
# from tasks.module1 import Module1Trainer
# from torch.utils.data import DataLoader


# def main(config_path):
#     # Load config
#     cfg = OmegaConf.load(config_path)

#     # Seed & device
#     torch.manual_seed(cfg.experiment.seed)
#     np.random.seed(cfg.experiment.seed)
#     device = cfg.experiment.device if torch.cuda.is_available() else "cpu"
#     print(f"Device: {device}")

#     # Load & group data
#     df_train = load_data(cfg.data.train_path)
#     df_val   = load_data(cfg.data.val_path)
#     df_test  = load_data(cfg.data.test_path)

#     samples_train = group_by_stock_date(df_train)
#     samples_val   = group_by_stock_date(df_val)
#     samples_test  = group_by_stock_date(df_test)

#     num_factors = (
#         cfg.model.num_factors
#         if cfg.model.num_factors is not None
#         else len(samples_train[0]["stock_factors"])
#     )
#     print(
#         f"num_factors={num_factors} | "
#         f"train={len(samples_train)} | val={len(samples_val)} | test={len(samples_test)}"
#     )

#     # Tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

#     # Datasets
#     use_mock = cfg.data.use_mock_srl
#     train_ds = NewsFactorDataset(samples_train, tokenizer, cfg.data.max_len, use_mock)
#     val_ds   = NewsFactorDataset(samples_val,   tokenizer, cfg.data.max_len, use_mock)
#     test_ds  = NewsFactorDataset(samples_test,  tokenizer, cfg.data.max_len, use_mock)

#     # DataLoaders
#     dl_cfg        = cfg.data.dataloader
#     train_sampler = make_sampler(samples_train)
#     train_loader  = DataLoader(
#         train_ds,
#         batch_size  = dl_cfg.batch_size,
#         sampler     = train_sampler,
#         num_workers = dl_cfg.num_workers,
#         pin_memory  = dl_cfg.pin_memory,
#         collate_fn  = collate_fn,
#     )
#     val_loader = DataLoader(
#         val_ds,
#         batch_size  = dl_cfg.batch_size,
#         shuffle     = False,
#         num_workers = dl_cfg.num_workers,
#         pin_memory  = dl_cfg.pin_memory,
#         collate_fn  = collate_fn,
#     )
#     test_loader = DataLoader(
#         test_ds,
#         batch_size  = dl_cfg.batch_size,
#         shuffle     = False,
#         num_workers = dl_cfg.num_workers,
#         pin_memory  = dl_cfg.pin_memory,
#         collate_fn  = collate_fn,
#     )

#     # Build model
#     cfg.model.num_factors = num_factors
#     model = build_model(cfg).to(device)

#     # Optimizer (differential LR)
#     bert_params    = list(model.roberta.parameters())
#     bert_param_ids = {id(p) for p in bert_params}
#     alpha_params   = [model.W_alpha]
#     alpha_ids      = {id(p) for p in alpha_params}
#     head_params    = [
#         p for p in model.parameters()
#         if id(p) not in bert_param_ids and id(p) not in alpha_ids
#     ]

#     opt_cfg   = cfg.optimizer
#     optimizer = torch.optim.AdamW([
#         {"params": bert_params,  "lr": opt_cfg.bert_lr,  "weight_decay": opt_cfg.weight_decay},
#         {"params": head_params,  "lr": opt_cfg.head_lr,  "weight_decay": opt_cfg.weight_decay},
#         {"params": alpha_params, "lr": opt_cfg.alpha_lr, "weight_decay": 0.0},
#     ])

#     # Scheduler (cosine with warmup)
#     train_cfg = cfg.training
#     total_ep  = train_cfg.epochs
#     warmup_ep = train_cfg.bert_warmup_epochs

#     def lr_lambda(epoch):
#         if epoch < warmup_ep:
#             return (epoch + 1) / warmup_ep
#         progress = (epoch - warmup_ep) / max(total_ep - warmup_ep, 1)
#         return 0.5 * (1 + np.cos(np.pi * progress))

#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

#     # ── Khởi tạo Trainer ──────────────────────────────────────────────────
#     trainer = Module1Trainer(model, optimizer, scheduler, cfg, device)

#     # Resume checkpoint
#     ckpt_cfg      = cfg.checkpoint
#     start_epoch   = 1
#     best_val_loss = float("inf")

#     if ckpt_cfg.resume and os.path.exists(ckpt_cfg.path):
#         start_epoch, best_val_loss = trainer.load_checkpoint(ckpt_cfg.path)
#         start_epoch += 1
#         print(f"Resuming from epoch {start_epoch}")

#     # BERT warm-up freeze
#     if start_epoch <= warmup_ep:
#         print(f"🔒 Freezing BERT for first {warmup_ep} epochs")
#         model.freeze_bert()

#     # ── Training loop ──────────────────────────────────────────────────────
#     patience_counter = 0

#     for epoch in range(start_epoch, total_ep + 1):
#         if epoch == warmup_ep + 1:
#             print(f"🔓 Unfreezing BERT at epoch {epoch}")
#             model.unfreeze_bert()

#         print(f"\nEpoch {epoch}/{total_ep}")

#         train_loss, train_acc = trainer.train_epoch(train_loader)
#         val_loss, val_acc, val_prec, val_rec, val_f1 = trainer.evaluate(val_loader)
#         scheduler.step()

#         # Alpha stats
#         if train_cfg.logging.log_alpha:
#             w_n, w_f = model.get_alpha_stats()
#             print(f"  W_alpha: news={w_n:.4f}  factor={w_f:.4f}  ratio={w_n/(w_f+1e-8):.3f}")

#         print(f"  train → loss={train_loss:.4f}  acc={train_acc:.4f}")
#         print(
#             f"  val   → loss={val_loss:.4f}  acc={val_acc:.4f}  "
#             f"prec={val_prec:.4f}  rec={val_rec:.4f}  f1={val_f1:.4f}"
#         )

#         if val_loss < best_val_loss:
#             best_val_loss    = val_loss
#             patience_counter = 0
#             torch.save(model.state_dict(), ckpt_cfg.save_best_path)
#             print(f"  ✅ Saved best model  (val_loss={val_loss:.4f})")
#         else:
#             patience_counter += 1
#             print(f"  No improvement ({patience_counter}/{train_cfg.patience})")

#         trainer.save_checkpoint(ckpt_cfg.path, epoch, best_val_loss)

#         if patience_counter >= train_cfg.patience:
#             print(f"\n⚠ Early stopping at epoch {epoch}.")
#             break

#     # ── Test evaluation ────────────────────────────────────────────────────
#     print("\n── Test evaluation ──")
#     model.load_state_dict(torch.load(ckpt_cfg.save_best_path, map_location=device))
#     test_loss, test_acc, test_prec, test_rec, test_f1 = trainer.evaluate(test_loader)
#     print(
#         f"Test → loss={test_loss:.4f}  acc={test_acc:.4f}  "
#         f"prec={test_prec:.4f}  rec={test_rec:.4f}  f1={test_f1:.4f}"
#     )

#     # ── Export features ────────────────────────────────────────────────────
#     if cfg.export.enable:
#         print("\n📤 Exporting features...")
#         trainer.export_features(train_loader, cfg.export.train_output)
#         trainer.export_features(val_loader,   cfg.export.val_output)
#         trainer.export_features(test_loader,  cfg.export.test_output)


# # if __name__ == "__main__":
# #     main()
    
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--config-file",
#         type=str,
#         required=True,
#         help="Path to config file"
#     )

#     args = parser.parse_args()

#     main(args.config_file)

# main.py
import os
import argparse
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from builders.model_builder import build_model
from data_utils.module1 import (
    load_data,
    group_by_stock_date,
    make_sampler,
    NewsFactorDataset,
    collate_fn,
)
from tasks.module1 import Module1Trainer
from torch.utils.data import DataLoader


def main(config_path: str):
    cfg = OmegaConf.load(config_path)

    torch.manual_seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)
    device = cfg.experiment.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    df_train = load_data(cfg.data.train_path)
    df_val   = load_data(cfg.data.val_path)
    df_test  = load_data(cfg.data.test_path)

    samples_train = group_by_stock_date(df_train)
    samples_val   = group_by_stock_date(df_val)
    samples_test  = group_by_stock_date(df_test)

    num_factors = (
        cfg.model.num_factors
        if cfg.model.num_factors is not None
        else len(samples_train[0]["stock_factors"])
    )
    print(
        f"num_factors={num_factors} | "
        f"train={len(samples_train)} | val={len(samples_val)} | test={len(samples_test)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

    # NewsFactorDataset không còn use_mock_srl
    train_ds = NewsFactorDataset(samples_train, tokenizer, cfg.data.max_len)
    val_ds   = NewsFactorDataset(samples_val,   tokenizer, cfg.data.max_len)
    test_ds  = NewsFactorDataset(samples_test,  tokenizer, cfg.data.max_len)

    dl_cfg        = cfg.data.dataloader
    train_sampler = make_sampler(samples_train)
    train_loader  = DataLoader(
        train_ds, batch_size=dl_cfg.batch_size, sampler=train_sampler,
        num_workers=dl_cfg.num_workers, pin_memory=dl_cfg.pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=dl_cfg.batch_size, shuffle=False,
        num_workers=dl_cfg.num_workers, pin_memory=dl_cfg.pin_memory,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=dl_cfg.batch_size, shuffle=False,
        num_workers=dl_cfg.num_workers, pin_memory=dl_cfg.pin_memory,
        collate_fn=collate_fn,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    cfg.model.num_factors = num_factors
    model = build_model(cfg).to(device)

    # ── Optimizer (differential LR) ───────────────────────────────────────
    bert_params    = list(model.roberta.parameters())
    bert_param_ids = {id(p) for p in bert_params}

    # FIX: W_news_logit và W_factor_logit thay vì W_alpha
    alpha_params = [model.W_news_logit, model.W_factor_logit]
    alpha_ids    = {id(p) for p in alpha_params}

    head_params  = [
        p for p in model.parameters()
        if id(p) not in bert_param_ids and id(p) not in alpha_ids
    ]

    opt_cfg   = cfg.optimizer
    optimizer = torch.optim.AdamW([
        {"params": bert_params,  "lr": opt_cfg.bert_lr,  "weight_decay": opt_cfg.weight_decay},
        {"params": head_params,  "lr": opt_cfg.head_lr,  "weight_decay": opt_cfg.weight_decay},
        {"params": alpha_params, "lr": opt_cfg.alpha_lr, "weight_decay": 0.0},
    ])

    # ── Scheduler ─────────────────────────────────────────────────────────
    train_cfg = cfg.training
    total_ep  = train_cfg.epochs
    warmup_ep = train_cfg.bert_warmup_epochs

    def lr_lambda(epoch):
        if epoch < warmup_ep:
            return (epoch + 1) / warmup_ep
        progress = (epoch - warmup_ep) / max(total_ep - warmup_ep, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = Module1Trainer(model, optimizer, scheduler, cfg, device)

    ckpt_cfg      = cfg.checkpoint
    start_epoch   = 1
    best_val_loss = float("inf")

    if ckpt_cfg.resume and os.path.exists(ckpt_cfg.path):
        start_epoch, best_val_loss = trainer.load_checkpoint(ckpt_cfg.path)
        start_epoch += 1
        print(f"Resuming from epoch {start_epoch}")

    if start_epoch <= warmup_ep:
        print(f"🔒 Freezing BERT for first {warmup_ep} epochs")
        model.freeze_bert()

    # ── Training loop ──────────────────────────────────────────────────────
    patience_counter = 0

    for epoch in range(start_epoch, total_ep + 1):
        if epoch == warmup_ep + 1:
            print(f"🔓 Unfreezing BERT at epoch {epoch}")
            model.unfreeze_bert()

        print(f"\nEpoch {epoch}/{total_ep}")
        train_loss, train_acc = trainer.train_epoch(train_loader)
        val_loss, val_acc, val_prec, val_rec, val_f1 = trainer.evaluate(val_loader)
        scheduler.step()

        if train_cfg.logging.log_alpha:
            w_n, w_f = model.get_alpha_stats()
            print(f"  W_alpha: news={w_n:.4f}  factor={w_f:.4f}  ratio={w_n/(w_f+1e-8):.3f}")

        print(f"  train → loss={train_loss:.4f}  acc={train_acc:.4f}")
        print(
            f"  val   → loss={val_loss:.4f}  acc={val_acc:.4f}  "
            f"prec={val_prec:.4f}  rec={val_rec:.4f}  f1={val_f1:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_cfg.save_best_path)
            print(f"  ✅ Saved best model  (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{train_cfg.patience})")

        trainer.save_checkpoint(ckpt_cfg.path, epoch, best_val_loss)

        if patience_counter >= train_cfg.patience:
            print(f"\n⚠ Early stopping at epoch {epoch}.")
            break

    # ── Test ──────────────────────────────────────────────────────────────
    print("\n── Test evaluation ──")
    model.load_state_dict(torch.load(ckpt_cfg.save_best_path, map_location=device))
    test_loss, test_acc, test_prec, test_rec, test_f1 = trainer.evaluate(test_loader)
    print(
        f"Test → loss={test_loss:.4f}  acc={test_acc:.4f}  "
        f"prec={test_prec:.4f}  rec={test_rec:.4f}  f1={test_f1:.4f}"
    )

    # ── Export ────────────────────────────────────────────────────────────
    if cfg.export.enable:
        print("\n📤 Exporting features...")
        trainer.export_features(train_loader, cfg.export.train_output)
        trainer.export_features(val_loader,   cfg.export.val_output)
        trainer.export_features(test_loader,  cfg.export.test_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    args = parser.parse_args()
    main(args.config_file)