# modules/module1.py
from __future__ import annotations

import warnings
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from builders.registry import register_model

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# SRL FEATURE EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════

class SRLFeatureExtractor(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def _pool_span(self, token_emb: torch.Tensor, indices: List[int]) -> torch.Tensor:
        """Mean-pool token embeddings tại các vị trí indices."""
        if not indices:
            return torch.zeros(self.hidden_size, device=token_emb.device)
        valid = [i for i in indices if i < token_emb.size(0)]
        if not valid:
            return torch.zeros(self.hidden_size, device=token_emb.device)
        return token_emb[valid].mean(dim=0)

    def forward(self, token_emb, srl_spans_batch, B, N_max):
        """
        token_emb      : (B*N_max, L, H) — output của RoBERTa
        srl_spans_batch: list[list[list[dict]]] — batch × news × predicates
        Trả về e_V, e_A0, e_A1, mỗi shape (B*N_max, H).
        """
        device = token_emb.device
        H = self.hidden_size
        e_V_list, e_A0_list, e_A1_list = [], [], []

        flat_idx = 0
        for b in range(B):
            for n in range(N_max):
                emb        = token_emb[flat_idx]   # (L, H)
                predicates = (
                    srl_spans_batch[b][n]
                    if n < len(srl_spans_batch[b]) else []
                )

                if not predicates:
                    e_V_list.append(torch.zeros(H, device=device))
                    e_A0_list.append(torch.zeros(H, device=device))
                    e_A1_list.append(torch.zeros(H, device=device))
                else:
                    v_vecs, a0_vecs, a1_vecs = [], [], []
                    for pred_dict in predicates:
                        v_vecs.append(self._pool_span(emb, pred_dict.get('V',  [])))
                        a0_vecs.append(self._pool_span(emb, pred_dict.get('A0', [])))
                        a1_vecs.append(self._pool_span(emb, pred_dict.get('A1', [])))
                    e_V_list.append(torch.stack(v_vecs).mean(0))
                    e_A0_list.append(torch.stack(a0_vecs).mean(0))
                    e_A1_list.append(torch.stack(a1_vecs).mean(0))

                flat_idx += 1

        return (
            torch.stack(e_V_list),    # (B*N_max, H)
            torch.stack(e_A0_list),
            torch.stack(e_A1_list),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SDPG EDGE FEATURE BUILDER
# FIX: mỗi loại edge có projection riêng để mô hình phân biệt được
#      quan hệ V-A0, V-A1 và A0-A1
# ═══════════════════════════════════════════════════════════════════════════════

# class SDPGEdgeBuilder(nn.Module):
#     def __init__(self, hidden_size: int):
#         super().__init__()
#         # 3 projections riêng biệt — mỗi cái học 1 loại quan hệ ngữ nghĩa khác nhau
#         self.proj_VA0  = nn.Linear(2 * hidden_size, hidden_size, bias=False)
#         self.proj_VA1  = nn.Linear(2 * hidden_size, hidden_size, bias=False)
#         self.proj_A0A1 = nn.Linear(2 * hidden_size, hidden_size, bias=False)

#     def forward(self, e_V, e_A0, e_A1):
#         """
#         Input shape: (B*N_max, H) mỗi tensor.
#         Output: G_VA0, G_VA1, G_A0A1, e_SDPG — mỗi cái (B*N_max, H).
#         """
#         G_VA0  = self.proj_VA0 (torch.cat([e_V,  e_A0], dim=-1))
#         G_VA1  = self.proj_VA1 (torch.cat([e_V,  e_A1], dim=-1))
#         G_A0A1 = self.proj_A0A1(torch.cat([e_A0, e_A1], dim=-1))
#         e_SDPG = torch.stack([G_VA0, G_VA1, G_A0A1], dim=1).mean(dim=1)
#         return G_VA0, G_VA1, G_A0A1, e_SDPG

class SDPGEdgeBuilder(nn.Module):
    """
    Paper: G_AB là edge feature giữa 2 roles trong semantic dependency graph.
    Dùng hadamard product (element-wise multiply) + projection để giữ
    tính đối xứng và đúng tinh thần dependency graph.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        # Projection sau hadamard để học ngữ nghĩa của từng loại edge
        self.proj_VA0  = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_VA1  = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_A0A1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.act = nn.Tanh()

    def forward(self, e_V, e_A0, e_A1):
        """
        e_V, e_A0, e_A1: (B*N_max, H)
        Hadamard product giữ đặc trưng tương tác, projection học edge type.
        """
        G_VA0  = self.act(self.proj_VA0 (e_V  * e_A0))   # (B*N_max, H)
        G_VA1  = self.act(self.proj_VA1 (e_V  * e_A1))
        G_A0A1 = self.act(self.proj_A0A1(e_A0 * e_A1))
        e_SDPG = (G_VA0 + G_VA1 + G_A0A1) / 3
        return G_VA0, G_VA1, G_A0A1, e_SDPG


# ═══════════════════════════════════════════════════════════════════════════════
# NEWS FACTORIZATION MODULE
# ═══════════════════════════════════════════════════════════════════════════════

@register_model("NewsFactorizationModule")
class NewsFactorizationModule(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.model_name      = config.model_name
        self.hidden_size     = config.hidden_size
        self.num_factors     = config.num_factors
        self.num_classes     = config.num_classes
        self.mlp_hidden      = config.mlp_hidden
        self.dropout         = config.dropout
        self.label_smoothing = getattr(config, "label_smoothing", 0.0)

        H = self.hidden_size
        F = self.num_factors

        self._news_feat_dim  = 6 * H
        self._total_feat_dim = 6 * H + F

        # ── Backbone ──────────────────────────────────────────────────────
        self.roberta = AutoModel.from_pretrained(self.model_name)

        # ── SRL + SDPG ────────────────────────────────────────────────────
        self.srl_extractor = SRLFeatureExtractor(H)
        self.sdpg_builder  = SDPGEdgeBuilder(H)     # FIX: 3 projections riêng

        # ── Factor normalization ──────────────────────────────────────────
        self.factor_norm = nn.LayerNorm(F)

        # ── W_alpha: scale riêng cho news part và factor part ─────────────
        # FIX: 2 scalar thay vì 1 vector (6H+F,) để tránh news (6H >> F) áp đảo
        # W_news  ∈ (0,1): scale chung cho toàn bộ news features
        # W_factor∈ (0,1): scale chung cho toàn bộ factor features
        # → gradient cân bằng hơn
        # self.W_news_logit   = nn.Parameter(torch.zeros(1))   # sigmoid → 0.5
        # self.W_factor_logit = nn.Parameter(torch.zeros(1))   # sigmoid → 0.5
        
        self.W_alpha = nn.Parameter(torch.ones(self._total_feat_dim))


        # Vẫn giữ W_alpha (6H+F,) để tương thích export / alpha_stats
        # nhưng không dùng trong forward nữa
        # (nếu cần per-dim gating thì uncomment phần dưới)

        # ── Classifier ────────────────────────────────────────────────────
        self.mlp = nn.Sequential(
            nn.Linear(self._total_feat_dim, self.mlp_hidden),
            nn.LayerNorm(self.mlp_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mlp_hidden, self.num_classes),
        )

    # ── BERT control ──────────────────────────────────────────────────────────

    def freeze_bert(self):
        for p in self.roberta.parameters():
            p.requires_grad = False

    def unfreeze_bert(self):
        for p in self.roberta.parameters():
            p.requires_grad = True

    # ── Alpha stats (for logging) ──────────────────────────────────────────────

    # def get_alpha_stats(self):
    #     w_news   = torch.sigmoid(self.W_news_logit).item()
    #     w_factor = torch.sigmoid(self.W_factor_logit).item()
    #     return w_news, w_factor
    
    def get_alpha_stats(self):
        w = torch.sigmoid(self.W_alpha)
        news_w   = w[:6*self.hidden_size].mean().item()
        factor_w = w[6*self.hidden_size:].mean().item()
        return news_w, factor_w

    # ── Build X (Eq. 4) ───────────────────────────────────────────────────────

    # def _build_X(self, x_cols, stock_factors, N_max):
    #     """
    #     Eq.(4): X = W_α ⊙ [X_n ; X_f]

    #     FIX: dùng 2 scalar gates riêng (W_news, W_factor) thay vì
    #          1 vector (6H+F,) để tránh gradient mất cân bằng khi 6H >> F.

    #     x_cols       : (B, N_max, 6H)
    #     stock_factors: (B, F)
    #     """
    #     B = x_cols.size(0)

    #     # Factor: LayerNorm → expand
    #     factors_normed = self.factor_norm(stock_factors)              # (B, F)
    #     X_f = factors_normed.unsqueeze(1).expand(B, N_max, self.num_factors)  # (B, N_max, F)

    #     # Gate
    #     w_n = torch.sigmoid(self.W_news_logit)    # scalar ∈ (0,1)
    #     w_f = torch.sigmoid(self.W_factor_logit)  # scalar ∈ (0,1)

    #     X = torch.cat([x_cols * w_n, X_f * w_f], dim=-1)   # (B, N_max, 6H+F)
    #     return X
    
    def _build_X(self, x_cols, stock_factors, N_max):
        B = x_cols.size(0)
        factors_normed = self.factor_norm(stock_factors)
        X_f = factors_normed.unsqueeze(1).expand(B, N_max, self.num_factors)
        
        X_cat = torch.cat([x_cols, X_f], dim=-1)          # (B, N_max, 6H+F)
        w = torch.sigmoid(self.W_alpha)                     # (6H+F,) per-dim gating
        X = X_cat * w.unsqueeze(0).unsqueeze(0)            # broadcast
        return X

    

    # ── Encoder (dùng cho cả forward và export) ────────────────────────────────

    def _encode(self, input_ids, attention_mask, srl_spans, news_counts, stock_factors):
        B, N_max, L = input_ids.shape
        H = self.hidden_size

        # RoBERTa
        flat_ids   = input_ids.view(B * N_max, L)
        flat_masks = attention_mask.view(B * N_max, L)
        token_emb  = self.roberta(
            input_ids=flat_ids, attention_mask=flat_masks
        ).last_hidden_state                                   # (B*N_max, L, H)

        # SRL → e_V, e_A0, e_A1  shape (B*N_max, H)
        e_V_flat, e_A0_flat, e_A1_flat = self.srl_extractor(
            token_emb, srl_spans, B, N_max
        )

        # SDPG → G_VA0, G_VA1, G_A0A1  shape (B*N_max, H)
        G_VA0_flat, G_VA1_flat, G_A0A1_flat, _ = self.sdpg_builder(
            e_V_flat, e_A0_flat, e_A1_flat
        )

        # Reshape → (B, N_max, H)
        def _r(t): return t.view(B, N_max, H)
        eV,    eA0,   eA1   = _r(e_V_flat),   _r(e_A0_flat),   _r(e_A1_flat)
        G_VA0, G_VA1, G_A0A1 = _r(G_VA0_flat), _r(G_VA1_flat), _r(G_A0A1_flat)
        e_SDPG = torch.stack([G_VA0, G_VA1, G_A0A1], dim=2).mean(dim=2)  # (B, N_max, H)

        # Build X (Eq. 4)
        x_cols = torch.cat([eV, eA0, eA1, G_VA0, G_VA1, G_A0A1], dim=-1)  # (B, N_max, 6H)
        X = self._build_X(x_cols, stock_factors, N_max)                     # (B, N_max, 6H+F)

        # Mask padding news
        mask = (
            torch.arange(N_max, device=X.device)
            .unsqueeze(0)
            .lt(news_counts.unsqueeze(1))
        ).unsqueeze(-1).float()                                              # (B, N_max, 1)

        # Eq. (5): y_n = Softmax(MLP(X[:, n]))  — per-news prediction
        # FIX: aggregate AFTER softmax (mean of probs), không phải mean of logits
        logits_per_news = self.mlp(X)                                        # (B, N_max, C)
        
        masked_logits = (logits_per_news * mask).sum(dim=1) \
                / mask.sum(dim=1).clamp(min=1) 
                
        probs  = F.softmax(masked_logits, dim=-1)                 # (B, N_max, C)

        # logits giả để dùng với CrossEntropyLoss (log của probs)
        # logits = torch.log(probs.clamp(min=1e-8))

        return {
            "logits":  masked_logits,  # (B, C) — dùng để tính loss
            "probs":   probs,
            "eV":      eV,
            "eA0":     eA0,
            "eA1":     eA1,
            "G_VA0":   G_VA0,
            "G_VA1":   G_VA1,
            "G_A0A1":  G_A0A1,
            "e_SDPG":  e_SDPG,
        }

    def forward(self, input_ids, attention_mask, srl_spans, news_counts, stock_factors):
        out = self._encode(input_ids, attention_mask, srl_spans, news_counts, stock_factors)
        return out["logits"], out["probs"]