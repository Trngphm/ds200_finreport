
# data_utils/module1.py
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import ast
from collections import defaultdict
from typing import List
import torch
import numpy as np
from features.srl import get_srl_backend


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING & GROUPING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['stock_factors'] = df['stock_factors'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date.astype(str)
    return df


def group_by_stock_date(df: pd.DataFrame) -> List[dict]:
    MAX_NEWS = 8
    groups = defaultdict(list)
    for _, row in df.iterrows():
        key = (str(row['CODE']), row['trade_date'])
        groups[key].append(row)

    samples = []
    for (code, trade_date), rows in groups.items():
        rows = rows[:MAX_NEWS]
        texts         = [r['text_a'] for r in rows]
        stock_factors = rows[0]['stock_factors']
        label         = int(rows[0]['label'])
        samples.append({
            'texts':         texts,
            'stock_factors': stock_factors,
            'label':         label,
            'code':          code,
            'trade_date':    trade_date,
        })
    return samples


def make_sampler(samples: List[dict]) -> WeightedRandomSampler:
    labels        = [s['label'] for s in samples]
    class_counts  = np.bincount(labels)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(sample_weights),
        replacement=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: word-level indices → char spans → subword token indices
# ═══════════════════════════════════════════════════════════════════════════════

def _words_to_char_spans(sentence: str, words: List[str]) -> List[tuple]:
    """
    Tính (char_start, char_end) cho từng từ trong `words` dựa trên `sentence`.
    Trả về list độ dài = len(words), mỗi phần tử là (cs, ce) [inclusive start,
    exclusive end] trong string gốc.

    Ví dụ:
        sentence = "华大基因中标河北"
        words    = ["华大基因", "中标", "河北"]
        → [(0,4), (4,6), (6,8)]
    """
    spans = []
    cursor = 0
    for word in words:
        # Tìm vị trí xuất hiện tiếp theo của word từ cursor trở đi
        pos = sentence.find(word, cursor)
        if pos == -1:
            # fallback: dùng cursor hiện tại và bước qua len(word) chars
            pos = cursor
        spans.append((pos, pos + len(word)))
        cursor = pos + len(word)
    return spans


def _word_indices_to_token_indices(
    word_indices: List[int],
    word_char_spans: List[tuple],
    token_offsets: List[tuple],
) -> List[int]:
    """
    Chuyển word-level indices → subword token indices.

    word_indices    : list of int, vị trí từ trong câu đã segment
    word_char_spans : (char_start, char_end) của mỗi từ (từ _words_to_char_spans)
    token_offsets   : (char_start, char_end) của mỗi subword token (từ tokenizer)

    Logic:
        - Gom char range của các words được chọn → tập char positions
        - Tìm subword tokens nào overlap với char range đó
    """
    # Gom tất cả char positions của các từ được chọn
    char_set = set()
    for wi in word_indices:
        if wi < len(word_char_spans):
            cs, ce = word_char_spans[wi]
            char_set.update(range(cs, ce))

    if not char_set:
        return [0]

    tok_idxs = []
    for tok_i, (cs, ce) in enumerate(token_offsets):
        if cs == ce == 0:          # [CLS], [SEP], padding
            continue
        # Overlap: token span [cs, ce) giao với char_set
        if any(cs <= c < ce for c in char_set):
            tok_idxs.append(tok_i)

    return tok_idxs if tok_idxs else [0]


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class NewsFactorDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length: int = 128):
        self.samples    = samples
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.srl        = get_srl_backend()   # không còn use_mock

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        all_input_ids, all_attn_masks, all_srl_spans = [], [], []

        for text in s['texts']:
            # ── Tokenize ──────────────────────────────────────────────────
            enc = self.tokenizer(
                text,
                max_length        = self.max_length,
                padding           = 'max_length',
                truncation        = True,
                return_tensors    = 'pt',
                return_offsets_mapping = True,
            )
            input_ids = enc['input_ids'].squeeze(0)       # (L,)
            attn_mask = enc['attention_mask'].squeeze(0)  # (L,)
            offsets   = enc['offset_mapping'].squeeze(0).tolist()  # [(cs,ce), ...]

            # ── SRL: word-level indices + segmented words ─────────────────
            # parse() trả về (predicate_roles_list, words)
            predicate_roles_list, words = self.srl.parse(text)

            # Tính char span của từng từ trong words
            word_char_spans = _words_to_char_spans(text, words)

            # ── Map word indices → subword token indices ───────────────────
            token_predicates = []
            for pred_roles in predicate_roles_list:
                token_roles = {}
                for role in ('V', 'A0', 'A1'):
                    word_idxs = pred_roles.get(role, [])
                    token_roles[role] = _word_indices_to_token_indices(
                        word_idxs, word_char_spans, offsets
                    )
                token_predicates.append(token_roles)

            all_input_ids.append(input_ids)
            all_attn_masks.append(attn_mask)
            all_srl_spans.append(token_predicates)

        return {
            'input_ids':      torch.stack(all_input_ids),
            'attention_mask': torch.stack(all_attn_masks),
            'srl_spans':      all_srl_spans,
            'stock_factors':  torch.tensor(s['stock_factors'], dtype=torch.float32),
            'label':          torch.tensor(s['label'],         dtype=torch.long),
            'code':           s['code'],
            'trade_date':     s['trade_date'],
            'texts':          s['texts'],
        }


def collate_fn(batch):
    max_N = max(b['input_ids'].size(0) for b in batch)
    L     = batch[0]['input_ids'].size(1)
    padded_ids, padded_masks, news_counts, srl_spans_batch = [], [], [], []

    for b in batch:
        N   = b['input_ids'].size(0)
        pad = max_N - N
        news_counts.append(N)
        padded_ids.append(
            torch.cat([b['input_ids'],
                       torch.zeros(pad, L, dtype=torch.long)], dim=0)
        )
        padded_masks.append(
            torch.cat([b['attention_mask'],
                       torch.zeros(pad, L, dtype=torch.long)], dim=0)
        )
        srl_spans_batch.append(b['srl_spans'] + [[] for _ in range(pad)])

    return {
        'input_ids':      torch.stack(padded_ids),
        'attention_mask': torch.stack(padded_masks),
        'srl_spans':      srl_spans_batch,
        'news_counts':    torch.tensor(news_counts, dtype=torch.long),
        'stock_factors':  torch.stack([b['stock_factors'] for b in batch]),
        'label':          torch.stack([b['label']          for b in batch]),
        'code':           [b['code']       for b in batch],
        'trade_date':     [b['trade_date'] for b in batch],
        'texts':          [b['texts']      for b in batch],
    }