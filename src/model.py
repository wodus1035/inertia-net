"""
model.py — Inertia-Net.

One shared STraTS-style triplet encoder maps a patient's recent physiology to a
representation z, one shared linear head maps z to the scalar readiness score s(z),
and every cohort-specific parameter is a cutpoint c[d, h]:

    logit[d, h] = tau * ( s(z) - c[d, h] )

Input (one sample = one anchor-day):
    trip  : (B, L, 3)   (time, channel index, normalized value) tokens
    valid : (B, L, 1)   1 for observed tokens, 0 for padding
    S     : (B, 2)      static features (age, sex), used to condition the pooling
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Attention pooling conditioned on the static features. V is zero-initialized, so
# training starts from the unconditioned pool.
class AttentionPool(nn.Module):
    def __init__(self, d_model, d_attn=None, cond_dim=0):
        super().__init__()
        d_attn = d_attn or max(8, d_model // 8)
        self.W = nn.Linear(d_model, d_attn)
        self.u = nn.Linear(d_attn, 1, bias=False)
        self.V = nn.Linear(cond_dim, d_attn, bias=False) if cond_dim else None
        if self.V is not None:
            nn.init.zeros_(self.V.weight)

    def forward(self, seq, return_weights=False, cond=None):
        h = self.W(seq)
        if self.V is not None and cond is not None:
            h = h + self.V(cond).unsqueeze(1)
        a = self.u(torch.tanh(h))                    # (B, T, 1)
        alpha = torch.softmax(a, dim=1)
        pooled = (alpha * seq).sum(dim=1)            # (B, d_model)
        return (pooled, alpha.squeeze(-1)) if return_weights else pooled


# Continuous Value Embedding: U tanh(Wx + b), embeds a scalar without binning.
class CVE(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        h = max(1, int(math.isqrt(d_model)))
        self.W = nn.Linear(1, h)
        self.U = nn.Linear(h, d_model, bias=False)

    def forward(self, x):                      # (B, L) -> (B, L, d)
        return self.U(torch.tanh(self.W(x.unsqueeze(-1))))


# Encoder over the (time, channel, value) triplets. Only observed cells become
# tokens, and the FiLM pair (gamma, beta) starts at the identity.
class StratsEncoder(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=4, num_layers=2, dropout=0.1,
                 fusion=True, cond_dim=0):
        super().__init__()
        self.feat = nn.Embedding(n_features + 1, d_model)   # last index = padding
        self.pad_id = n_features
        self.cve_t = CVE(d_model)
        self.cve_v = CVE(d_model)
        self.fusion = fusion
        if fusion:
            self.gamma = nn.Embedding(n_features + 1, d_model)
            self.beta = nn.Embedding(n_features + 1, d_model)
            nn.init.ones_(self.gamma.weight)
            nn.init.zeros_(self.beta.weight)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool = AttentionPool(d_model, cond_dim=cond_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trip, valid, return_weights=False, cond=None):
        t, f, v = trip[..., 0], trip[..., 1], trip[..., 2]
        fid = torch.where(valid > 0, f, torch.full_like(f, float(self.pad_id))).long()
        ev = self.cve_v(v)
        if self.fusion:
            ev = self.gamma(fid) * ev + self.beta(fid)
        e = self.feat(fid) + self.cve_t(t) + ev
        e = self.dropout(e)
        pad = valid <= 0
        # a fully padded row would make the softmax NaN, so keep its first token
        pad = pad & ~(pad.all(dim=1, keepdim=True) & (torch.arange(pad.size(1), device=pad.device) == 0))
        c = self.transformer(e, src_key_padding_mask=pad)
        c = c.masked_fill(pad.unsqueeze(-1), 0.0)
        pooled, alpha = self.pool(c, return_weights=True, cond=cond)
        return (pooled, alpha) if return_weights else pooled


# One shared score plus per-cohort, per-horizon cutpoints, parameterized so that
# c[d,1] > c[d,2] > c[d,3] holds by construction. shared_cutpoints=True is the
# one-set-for-both ablation.
class InertiaHead(nn.Module):
    def __init__(self, d_model, n_horizons, domains, shared_cutpoints=False):
        super().__init__()
        self.n_horizons = n_horizons
        self.domains = list(domains)
        self.shared_cutpoints = bool(shared_cutpoints)
        self._cut_domains = ["shared"] if self.shared_cutpoints else self.domains
        self.score_head = nn.Linear(d_model, 1)  # shared across cohorts
        self.base = nn.ParameterDict({dom: nn.Parameter(torch.zeros(1)) for dom in self._cut_domains})
        if n_horizons > 1:
            self.steps = nn.ParameterDict(
                {dom: nn.Parameter(torch.zeros(n_horizons - 1)) for dom in self._cut_domains}
            )
        else:
            self.steps = None

    def cutpoints(self, domain):
        """(n_horizons,) cutpoints for `domain`, strictly decreasing."""
        if self.shared_cutpoints:
            domain = "shared"
        b0 = self.base[domain]
        if self.n_horizons == 1:
            return b0
        pos_steps = F.softplus(self.steps[domain])
        cuts, cur = [b0], b0
        for i in range(self.n_horizons - 1):
            cur = cur - pos_steps[i]
            cuts.append(cur)
        return torch.cat(cuts)

    def forward(self, fused, domain):
        return self.score_head(fused) - self.cutpoints(domain).unsqueeze(0)


# Encoder wrapper; the name is kept because checkpoints and analysis scripts
# refer to it. forward() returns (None, z) and InertiaHead is applied to z.
class MultiModalTransformer(nn.Module):
    def __init__(self, vital_input_dim, static_input_dim=2, d_model=64, nhead=4,
                 num_layers=2, dropout=0.2, n_horizons=3, logit_scale=3.0,
                 strats_fusion=True, **_ignored):
        super().__init__()
        self.logit_scale = logit_scale
        self.n_horizons = n_horizons
        self.strats = StratsEncoder(vital_input_dim, d_model, nhead, num_layers, dropout,
                                    fusion=strats_fusion, cond_dim=static_input_dim)

    def forward(self, trip, valid, S, return_fused=True):
        """trip (B,L,3), valid (B,L,1), S (B,static). Returns (None, z)."""
        fused = self.strats(trip, valid[..., 0], cond=S)
        return None, fused


# Class-weighted BCE; the paper uses smoothing=0.
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=2.5, smoothing=0.0):
        super().__init__()
        self.smoothing = smoothing
        pw = torch.as_tensor(pos_weight, dtype=torch.float32)
        if pw.ndim == 0:
            pw = pw.unsqueeze(0)
        self.register_buffer('pos_weight', pw)

    def forward(self, logits, targets):
        target_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(
            logits, target_smooth, pos_weight=self.pos_weight
        )


# Multi-kernel Gaussian MMD between the two cohorts' representations z.
def _pairwise_sq_dists(x, y):
    x2 = (x ** 2).sum(dim=1, keepdim=True)
    y2 = (y ** 2).sum(dim=1, keepdim=True)
    xy = x @ y.t()
    return (x2 + y2.t() - 2 * xy).clamp(min=0.0)


def mmd_loss(x, y, sigmas=(1.0, 2.0, 4.0, 8.0, 16.0)):
    """Biased multi-kernel MMD^2 estimator between two samples (x: (n,d), y: (m,d))."""
    xx = _pairwise_sq_dists(x, x)
    yy = _pairwise_sq_dists(y, y)
    xy = _pairwise_sq_dists(x, y)
    kxx, kyy, kxy = torch.zeros_like(xx), torch.zeros_like(yy), torch.zeros_like(xy)
    for sigma in sigmas:
        gamma = 1.0 / (2.0 * sigma ** 2)
        kxx = kxx + torch.exp(-gamma * xx)
        kyy = kyy + torch.exp(-gamma * yy)
        kxy = kxy + torch.exp(-gamma * xy)
    return kxx.mean() + kyy.mean() - 2 * kxy.mean()


# Post-hoc temperature scaling (Guo et al., 2017), fitted on the validation fold.
class TemperatureScaler(nn.Module):
    def __init__(self, n_horizons=1):
        super().__init__()
        self.n_horizons = n_horizons
        self.log_temperature = nn.Parameter(torch.zeros(n_horizons))

    @property
    def temperature(self):
        return torch.exp(self.log_temperature)

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, logits, targets, max_iter=200, lr=0.01):
        """logits, targets: (N, n_horizons) float tensors from a held-out split."""
        logits = logits.detach().float()
        targets = targets.detach().float()
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(logits / self.temperature, targets)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self.temperature.detach().cpu().numpy().tolist()
