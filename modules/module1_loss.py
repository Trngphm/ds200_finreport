
# modules/module1_loss.py
import torch


# def alpha_diversity_loss(model, lambda_div: float, lambda_entropy: float) -> torch.Tensor:
#     """
#     Diversity regularization cho W_news và W_factor.

#     L_div =
#         lambda_div     * (w_news - w_factor)^2       ← giữ cân bằng
#       - lambda_entropy * H([w_news, w_factor])        ← tránh collapse về 0 hoặc 1

#     Dùng 2 scalar gates (W_news_logit, W_factor_logit) thay vì vector (6H+F,)
#     để tính nhanh hơn và gradient cân bằng hơn.
#     """
#     w_news   = torch.sigmoid(model.W_news_logit)    # scalar
#     w_factor = torch.sigmoid(model.W_factor_logit)  # scalar

#     # Balance: tránh 1 bên dominate
#     balance_loss = (w_news - w_factor) ** 2

#     # Entropy: tránh collapse
#     eps = 1e-6
#     def _h(w):
#         w = w.clamp(eps, 1 - eps)
#         return -(w * w.log() + (1 - w) * (1 - w).log())

#     entropy_loss = (_h(w_news) + _h(w_factor)).mean()

#     return lambda_div * balance_loss - lambda_entropy * entropy_loss

# modules/module1_loss.py
def alpha_diversity_loss(model, lambda_div: float, lambda_entropy: float) -> torch.Tensor:
    w = torch.sigmoid(model.W_alpha)
    H = model.hidden_size
    F = model.num_factors
    
    w_news   = w[:6*H].mean()
    w_factor = w[6*H:].mean()
    
    # Balance loss
    balance_loss = (w_news - w_factor) ** 2
    
    # Entropy: tránh collapse về 0 hoặc 1 (tính trên toàn vector)
    eps = 1e-6
    w_c = w.clamp(eps, 1 - eps)
    entropy = -(w_c * w_c.log() + (1-w_c) * (1-w_c).log()).mean()
    
    return lambda_div * balance_loss - lambda_entropy * entropy