# features/srl.py
import warnings
from typing import List, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# SRL BACKEND  (HanLP — Chinese SRL)
# ═══════════════════════════════════════════════════════════════════════════════

class HanLPSRLBackend:
    """
    Wrapper quanh HanLP CPB3 SRL pipeline.

    HanLP trả về word-level indices (vị trí từ trong câu đã word-segment),
    KHÔNG phải character-position trong string gốc.
    Ta cần map: word_idx → char_span → subword token indices (BPE/WordPiece).
    Bước map này được thực hiện trong NewsFactorDataset.__getitem__ bằng
    cách dùng offsets_mapping của tokenizer.

    parse() trả về List[dict] với key = role ('V','A0','A1'),
    value = List[int] là WORD indices (0-based) trong câu đã segment.
    """

    _instance: Optional["HanLPSRLBackend"] = None

    def __init__(self):
        self._pipeline  = None
        self._segmenter = None
        self._load()

    def _load(self):
        try:
            import hanlp
            # Word segmenter — cần để biết char span của từng từ
            self._segmenter = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
            # SRL model
            self._pipeline  = hanlp.load(hanlp.pretrained.srl.CPB3_SRL_ELECTRA_SMALL)
            print("[SRL] HanLP loaded (segmenter + SRL).")
        except Exception as e:
            raise RuntimeError(
                f"[SRL] Cannot load HanLP: {e}\n"
                "      Install with: pip install hanlp"
            )

    def parse(self, sentence: str) -> List[dict]:
        """
        Trả về list các predicate dict.
        Mỗi dict: { 'V': [word_idx, ...], 'A0': [...], 'A1': [...] }
        word_idx là vị trí từ trong câu đã segment (0-based).
        Char span của từ word_idx được tính trong segment_to_char_spans().
        """
        try:
            words = self._segmenter(sentence)          # ['华大基因', '中标', ...]
            result = self._pipeline(words)             # list of predicate structures

            all_predicate_roles = []
            for pred in result:
                roles = {'V': [], 'A0': [], 'A1': []}
                for word_idx, (token, role) in enumerate(pred):
                    if role in roles:
                        roles[role].append(word_idx)
                if roles['V']:
                    all_predicate_roles.append(roles)

            if not all_predicate_roles:
                all_predicate_roles = [self._fallback(len(words))]

            # Đính kèm words để Dataset có thể tính char spans
            return all_predicate_roles, words

        except Exception as e:
            warnings.warn(f"[SRL] parse failed: {e}. Using fallback.")
            # fallback: trả về 1 predicate chia đều câu
            n_chars = len(sentence)
            words_fallback = list(sentence)            # mỗi char là 1 "từ"
            return [self._fallback(len(words_fallback))], words_fallback

    @staticmethod
    def _fallback(n_words: int) -> dict:
        """Chia đều n_words thành 3 phần V / A0 / A1."""
        third = max(n_words // 3, 1)
        return {
            'V':  list(range(third, 2 * third)),
            'A0': list(range(0, third)),
            'A1': list(range(2 * third, n_words)),
        }


# ── Singleton ──────────────────────────────────────────────────────────────────
_SRL_BACKEND: Optional[HanLPSRLBackend] = None

def get_srl_backend() -> HanLPSRLBackend:
    global _SRL_BACKEND
    if _SRL_BACKEND is None:
        _SRL_BACKEND = HanLPSRLBackend()
    return _SRL_BACKEND