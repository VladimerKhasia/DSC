#@title experiment
import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import logging
import requests
import tiktoken
import matplotlib.pyplot as plt
import random
import gc
import zipfile
import io
import numpy as np

# ==========================================
# 0. Hardware & System Setup
# ==========================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)

print("=== Hardware Status ===")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability()
    print(f" > GPU Detected: {gpu_name} (CC {major}.{minor})")
    torch.set_float32_matmul_precision('high')
else:
    print(" ! No GPU. Training will be slow.")

@dataclass
class Config:
    # --- EXPERIMENT CONTROL ---
    target_total_params: float = 35.0
    target_active_params: float = 28.0
    method: str = 'dsc'  # options: 'dsc', 'standard_moe', 'dense'

    d_model: int = 384
    n_layer: int = 6
    n_head: int = 6
    vocab_size: int = 50304
    block_size: int = 256
    
    dropout: float = 0.1 
    tie_weights: bool = True

    # Auto-Solver fields
    dense_d_ffn: int = 0
    num_experts: int = 0
    moe_top_k: int = 2
    expert_d_ffn: int = 0

    # --- DSC PARAMS ---
    dmc_top_k: int = 4
    num_bases: int = 0         
    dmc_base_d_ffn: int = 0    
    max_bases_limit: int = 512 

    # Equation Parameters
    tau_clamp: float = 20.0       
    target_budget_mu: float = 1.0 

    # --- MEMORY STRATEGY ---
    # [OPTIMIZATION] Increased Global Batch Size for stable routing gradients
    global_batch_size: int = 128 
    micro_batch_size: int = 16

    learning_rate: float = 6e-4 
    min_lr: float = 6e-5
    weight_decay: float = 0.02
    warmup_iters: int = 150
    max_iters: int = 2000
    eval_interval: int = 250
    eval_iters: int = 50
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- REGULARIZATION ---
    lambda_aux: float = 0.01     
    lambda_budget: float = 0.01  
    lambda_coh: float = 0.001
    lambda_z: float = 0.0001    # Router Z-Loss weight

# ==========================================
# 1. Automatic Fairness Solver
# ==========================================
def auto_configure_model(cfg: Config):
    """
    Solves for architecture dimensions to maintain Fair Compute.
    """
    print(f"\n[Auto-Config] Solving architecture for Target: {cfg.target_total_params}M Total / {cfg.target_active_params}M Active")

    fixed_params = (
        (cfg.vocab_size * cfg.d_model) +
        (cfg.block_size * cfg.d_model) +
        (cfg.n_layer * 4 * cfg.d_model**2) +
        (cfg.n_layer * 4 * cfg.d_model) +
        (2 * cfg.d_model)
    )

    if not cfg.tie_weights:
        fixed_params += (cfg.vocab_size * cfg.d_model)

    budget_total = (cfg.target_total_params * 1e6) - fixed_params
    budget_active = (cfg.target_active_params * 1e6) - fixed_params

    if budget_active <= 0:
        raise ValueError("Targets too small! Increase Target Active Params.")

    C_tot = budget_total / cfg.n_layer
    C_act = budget_active / cfg.n_layer
    D = cfg.d_model

    if cfg.method == 'dense':
        hidden_dim = int(budget_total / cfg.n_layer / (2 * D))
        cfg.dense_d_ffn = hidden_dim
        print(f" > [Dense] Width: {hidden_dim}")

    elif cfg.method == 'standard_moe':
        best_N = cfg.moe_top_k + 1
        best_H = 32
        min_error = float('inf')

        # [UPDATED] Search wider range for experts
        for N in range(cfg.moe_top_k + 1, 256):
            router_cost = D * N
            available_active = C_act - router_cost
            if available_active < (2 * D * 32 * cfg.moe_top_k): continue

            # H based on active budget
            H = int(available_active / (2 * D * cfg.moe_top_k))
            
            # Check total budget
            actual_tot = (N * 2 * D * H) + (D * N)
            error = abs(actual_tot - C_tot)

            if error < min_error:
                min_error = error
                best_N = N
                best_H = H

        cfg.num_experts = best_N
        cfg.expert_d_ffn = best_H
        print(f" > [MoE] Solved: Experts={best_N}, Hidden={best_H}")

    elif cfg.method == 'dsc':
            # [Algebraic Solver]
            # We solve a system of equations to hit both Total and Active targets EXACTLY.
            # Eq 1 (Total): Cost_Tot = 2*D*H + 3*D*M  (Static + Router + U + V)
            # Eq 2 (Active): Cost_Act = 2*D*H + 1*D*M + 2*D*K (Static + Router + K active bases)
            
            # Subtract Eq 2 from Eq 1:
            # Cost_Tot - Cost_Act = 2*D*M - 2*D*K
            # 2*D*M = Cost_Tot - Cost_Act + 2*D*K
            # M = (Cost_Tot - Cost_Act + 2*D*K) / (2*D)
            
            # Calculate M (Num Bases)
            numerator_M = C_tot - C_act + (2 * D * cfg.dmc_top_k)
            M = int(numerator_M / (2 * D))
            
            # Calculate H (Static Base Dim) using Eq 2:
            # 2*D*H = Cost_Act - D*M - 2*D*K
            cost_router = D * M
            cost_active_bases = 2 * D * cfg.dmc_top_k
            
            remaining_for_static = C_act - cost_router - cost_active_bases
            d_base = int(remaining_for_static / (2 * D))

            # [Safety Valve] 
            # If the Total Budget is huge but Active is small, M becomes huge, 
            # and the Router (D*M) might eat the entire Active budget, leaving d_base < 0.
            if d_base < 32:
                print(" ! [Warning] Total parameter target is too high relative to Active target.")
                print(" ! Clamping d_base to 32 and reducing Bases (M) to maintain Active Fairness.")
                d_base = 32
                # Recalculate M based on Active Constraint only (ignoring Total Constraint to remain fair)
                # C_act = 2*D*H + D*M + 2*D*K  =>  D*M = C_act - 2*D*H - 2*D*K
                fixed_active_cost = (2 * D * d_base) + (2 * D * cfg.dmc_top_k)
                M = int((C_act - fixed_active_cost) / D)

            cfg.num_bases = M
            cfg.dmc_base_d_ffn = d_base
            
            print(f" > [DSC] Algebraic Solution: Bases={M}, BaseDim={d_base}")
            print(f" > [DSC] Precision Check: Router Cost in Active={(D*M)/1e6:.2f}M")
            
    return cfg


# ==========================================
# 2. Data Loader
# ==========================================
class DataLoaderRigorous:
    def __init__(self, config, split='train', seed=42):
        self.B = config.micro_batch_size
        self.T = config.block_size
        self.device = config.device
        self.split = split
        random.seed(seed)

        filename = 'wikitext-103-raw-v1.zip'
        if not os.path.exists(filename):
            print(" > Downloading WikiText-103 Raw...")
            url = "https://huggingface.co/datasets/mattdangerw/wikitext-103-raw/resolve/main/wikitext-103-raw-v1.zip"
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                r = requests.get(url, stream=True, headers=headers)
                if r.status_code != 200: raise ValueError(f"Download failed: {r.status_code}")
                with open(filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024*1024): f.write(chunk)
            except Exception as e:
                print(f" ! Download failed: {e}")
                raise

        with zipfile.ZipFile(filename, 'r') as zip_ref:
            target_file = 'wiki.train.raw' if split == 'train' else 'wiki.valid.raw'
            found_path = None
            for name in zip_ref.namelist():
                if name.endswith(target_file):
                    found_path = name
                    break
            with zip_ref.open(found_path) as f:
                text_data = f.read(20_000_000).decode('utf-8')

        lines = text_data.split('\n')
        lines = [l for l in lines if len(l) > 10]

        augmented_lines = []
        for line in lines:
            augmented_lines.append(line)
            if random.random() < 0.15:
                a, b, c = random.randint(0,99), random.randint(0,99), random.randint(0,99)
                res = a + b + c
                eq = f"Calc: {a} + {b} + {c} = {res}"
                augmented_lines.append(eq)

        mixed_text = '\n'.join(augmented_lines)
        enc = tiktoken.get_encoding('gpt2')
        all_tokens = torch.tensor(enc.encode(mixed_text, allowed_special={'<|endoftext|>'}), dtype=torch.long)

        n = int(0.9 * len(all_tokens))
        self.data = all_tokens[:n] if split == 'train' else all_tokens[n:]
        print(f" > [{split}] Loaded {len(self.data)} tokens.")

    def next_batch(self):
        ix = torch.randint(len(self.data) - self.T, (self.B,))
        x = torch.stack([self.data[i:i+self.T] for i in ix])
        y = torch.stack([self.data[i+1:i+1+self.T] for i in ix])
        return x.contiguous().to(self.device), y.contiguous().to(self.device)

    def get_val_batch(self, idx):
        ptr = (idx * self.B * self.T) % (len(self.data) - self.B * self.T - 1)
        x = torch.stack([self.data[ptr + i*self.T : ptr + (i+1)*self.T] for i in range(self.B)])
        y = torch.stack([self.data[ptr + i*self.T + 1 : ptr + (i+1)*self.T + 1] for i in range(self.B)])
        return x.contiguous().to(self.device), y.contiguous().to(self.device)

# ==========================================
# 3. Model Classes (Highly Optimized)
# ==========================================

class DSCLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.num_bases = config.num_bases
        self.top_k = config.dmc_top_k
        self.static_dim = config.dmc_base_d_ffn

        self.tau = config.tau_clamp
        self.eps = 1e-6
        self.mu = config.target_budget_mu

        self.l_aux = config.lambda_aux
        self.l_budget = config.lambda_budget
        self.l_coh = config.lambda_coh
        self.l_z = config.lambda_z

        # 1. Static Backbone
        self.static_net = nn.Sequential(
            nn.Linear(self.d_model, self.static_dim, bias=False),
            nn.GELU(),
            nn.Linear(self.static_dim, self.d_model, bias=False)
        )

        # 2. Dynamic Router
        # [OPTIMIZATION] Input Norm for Router stability
        self.router_norm = nn.LayerNorm(self.d_model)
        self.router = nn.Linear(self.d_model, self.num_bases, bias=True)

        # 3. Basis Bank (U, V) - Orthogonal Init
        self.raw_U = nn.Parameter(torch.empty(self.num_bases, self.d_model))
        self.raw_V = nn.Parameter(torch.empty(self.num_bases, self.d_model))
        nn.init.orthogonal_(self.raw_U)
        nn.init.orthogonal_(self.raw_V)
        with torch.no_grad():
            self.raw_U.mul_(0.02)
            self.raw_V.mul_(0.02)
        
        # 4. Global Scale (Gamma) 
        # [OPTIMIZATION] Vector-wise Gamma for per-channel scaling
        self.gamma = nn.Parameter(torch.ones(1, 1, self.d_model) * 0.1)

        self.dropout = nn.Dropout(config.dropout)
        self.running_losses = {}

        # Init Router
        nn.init.normal_(self.router.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.router.bias)

    def _projected_norm(self, W):
        norms = torch.norm(W, p=2, dim=1, keepdim=True)
        return W / torch.clamp(norms, min=self.eps)

    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.view(-1, D)

        # --- 1. Routing (Stabilized) ---
        # Normalize input to router to decouple magnitude from direction
        router_input = self.router_norm(x_flat)
        r_raw = self.router(router_input)
        
        r = torch.clamp(r_raw, -self.tau, self.tau)
        alpha = F.softplus(r)

        # Top-K
        phi, indices = torch.topk(alpha, self.top_k, dim=-1)
        S_sum = phi.sum(dim=-1, keepdim=True)

        # --- Eq 4: Magnitude-Gated Quasi-Simplex ---
        direction = phi / (S_sum + self.eps)
        magnitude = torch.tanh(S_sum)
        Z_vals = direction * magnitude

        # --- 2. Retrieval ---
        U_norm = self._projected_norm(self.raw_U)
        V_norm = self._projected_norm(self.raw_V)

        U_selected = F.embedding(indices, U_norm)
        V_selected = F.embedding(indices, V_norm)

        # --- 3. Factorized Contraction ---
        h_lat = torch.einsum('bd,bkd->bk', x_flat, U_selected)
        
        # [OPTIMIZATION] Vector-wise Gamma broadcasting
        # h_lat: (B*S, K) -> needs reshaping for broadcasting if we did complex stuff, 
        # but here we apply gamma at the end.
        
        # Re-weight interaction
        h_weighted = h_lat * Z_vals
        
        # Expand
        dyn_out_flat = torch.einsum('bk,bkd->bd', h_weighted, V_selected)
        
        # Apply Vector Gamma (reshaped for broadcast)
        dyn_out = dyn_out_flat.view(B, S, D) * self.gamma

        # --- 4. Static Path ---
        static_out = self.static_net(x_flat).view(B, S, D)

        # --- 5. Regularization ---
        if self.training:
            probs = F.softmax(r, dim=-1)
            P_j = probs.mean(dim=0)
            L_aux = self.num_bases * torch.sum(P_j ** 2)

            avg_S = S_sum.mean()
            L_budget = (F.relu(self.mu - avg_S)) ** 2

            # [OPTIMIZATION] Z-Loss (stabilizes logits)
            # Penalize large positive logits to prevent exp() explosion
            log_z = torch.logsumexp(r_raw, dim=-1)
            L_z = torch.mean(log_z ** 2)

            def frame_potential(mat):
                gram = torch.mm(mat, mat.t())
                eye = torch.eye(mat.shape[0], device=mat.device)
                off_diag = gram * (1 - eye)
                return torch.norm(off_diag, p='fro') ** 2

            L_coh = frame_potential(U_norm) + frame_potential(V_norm)

            self.running_losses = {
                'aux': L_aux * self.l_aux,
                'budget': L_budget * self.l_budget,
                'coh': L_coh * self.l_coh,
                'z_loss': L_z * self.l_z
            }
        else:
            self.running_losses = {}

        return self.dropout(static_out + dyn_out)


class StandardMoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.moe_top_k
        self.d_model = config.d_model
        hidden_dim = config.expert_d_ffn
        
        # [FAIRNESS] Add Router Norm (same as DSC)
        self.router_norm = nn.LayerNorm(config.d_model)

        self.gate = nn.Linear(config.d_model, config.num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.d_model, hidden_dim, bias=False),
                nn.GELU(),
                nn.Linear(hidden_dim, config.d_model, bias=False)
            ) for _ in range(self.num_experts)
        ])
        self.dropout = nn.Dropout(config.dropout)
        self.running_losses = {}
        
        # [FAIRNESS] Use same Lambda Z as DSC
        self.l_z = getattr(config, 'lambda_z', 0.0001)

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)
        
        # [FAIRNESS] Apply Router Norm
        router_input = self.router_norm(x_flat)
        router_logits = self.gate(router_input)

        if self.training:
            # Jitter (Standard MoE technique)
            router_logits = router_logits + torch.randn_like(router_logits) * 0.01

        probs = F.softmax(router_logits, dim=1)
        importance = probs.sum(0)

        routing_weights, selected_experts = torch.topk(probs, self.top_k, dim=-1)
        
        load_mask = F.one_hot(selected_experts, num_classes=self.num_experts).float()
        load = load_mask.sum(1).sum(0) 

        scale = self.num_experts / (x_flat.shape[0] * x_flat.shape[0])
        aux_loss = torch.sum(importance * load) * scale
        
        # [FAIRNESS] Calculate Z-Loss for MoE
        # log_z = log(sum(exp(logits)))
        log_z = torch.logsumexp(router_logits, dim=-1)
        z_loss = torch.mean(log_z ** 2)
        
        if self.training:
            self.running_losses = {
                'moe_aux': aux_loss * 0.01,
                'moe_z': z_loss * self.l_z  # Add Z-loss here
            }
        else:
            self.running_losses = {}

        final_output = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_idx = selected_experts[:, i]
            weight = routing_weights[:, i].unsqueeze(1)
            for e_idx in range(self.num_experts):
                mask = (expert_idx == e_idx)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e_idx](expert_input)
                    final_output[mask] += expert_output * weight[mask]
        return self.dropout(final_output.view(B, T, C))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.dense_d_ffn, bias=False),
            nn.GELU(),
            nn.Linear(config.dense_d_ffn, config.d_model, bias=False),
            nn.Dropout(config.dropout)
        )
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        if config.method == 'dsc': self.ffn = DSCLayer(config)
        elif config.method == 'standard_moe': self.ffn = StandardMoELayer(config)
        else: self.ffn = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.n_head = config.n_head
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.resid_dropout = nn.Dropout(config.dropout)
        self.attn_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = self.attn_dropout(F.softmax(att, dim=-1))
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            self.head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

        # Scale residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('base_down.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
            if 'experts' in pn and pn.endswith('2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        router_params = []
        decay_params = []
        nodecay_params = []

        for pn, p in self.named_parameters():
            if not p.requires_grad: continue
            
            if 'router' in pn or 'gate' in pn:
                router_params.append(p)
            elif p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)

        print(f" > Optimizer: {len(decay_params)} decay, {len(nodecay_params)} no-decay, {len(router_params)} routers (High LR).")

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate},
            {'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate},
            {'params': router_params, 'weight_decay': 0.0, 'lr': learning_rate * 5.0}
        ]

        use_fused = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
        extra_args = dict(fused=True) if use_fused and device_type == 'cuda' else dict()
        optimizer = torch.optim.AdamW(optim_groups, betas=(0.9, 0.95), **extra_args)
        return optimizer

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=idx.device))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# ==========================================
# 5. Runner
# ==========================================

def get_lr(it, cfg):
    if it < cfg.warmup_iters:
        return cfg.learning_rate * it / cfg.warmup_iters
    if it > cfg.max_iters:
        return cfg.min_lr
    decay_ratio = (it - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, eval_iters):
    out = {}
    val_latency_accum = 0.0
    val_batches = 0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    model.eval()
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            if split == 'val': X, Y = loader.get_val_batch(k)
            else: X, Y = loader.next_batch()

            if split == 'val':
                torch.cuda.synchronize()
                start_event.record()

            with torch.amp.autocast('cuda'):
                logits, loss = model(X, Y)

            if split == 'val':
                end_event.record()
                torch.cuda.synchronize()
                if k > 0:
                    val_latency_accum += start_event.elapsed_time(end_event)
                    val_batches += 1

            losses[k] = loss.item()
        out[split] = losses.mean().item()

    model.train()
    avg_latency_ms = (val_latency_accum / val_batches) if val_batches > 0 else 0.0
    return out, avg_latency_ms

def run_experiment(method_name, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    cfg = Config(method=method_name)
    cfg = auto_configure_model(cfg)

    # [OPTIMIZATION] Accumulation for Gradient Hygiene
    assert cfg.global_batch_size % cfg.micro_batch_size == 0
    grad_accum_steps = cfg.global_batch_size // cfg.micro_batch_size
    print(f" > [Memory Strategy] Global BS={cfg.global_batch_size}, Micro BS={cfg.micro_batch_size}, Accum Steps={grad_accum_steps}")

    train_loader = DataLoaderRigorous(cfg, 'train', seed)
    val_loader = DataLoaderRigorous(cfg, 'val', seed)

    model = GPT(cfg).to(cfg.device)
    optimizer = model.configure_optimizers(cfg.weight_decay, cfg.learning_rate, cfg.device)
    scaler = torch.amp.GradScaler('cuda')

    history = {'iter': [], 'val_loss': [], 'train_loss': []}
    final_latency = 0.0

    optimizer.zero_grad(set_to_none=True)

    try:
        for iter in range(cfg.max_iters + 1):
            lr = get_lr(iter, cfg)
            
            for param_group in optimizer.param_groups:
                if param_group['lr'] > lr * 2.0: # Router group
                     param_group['lr'] = lr * 5.0
                else:
                     param_group['lr'] = lr

            if iter % cfg.eval_interval == 0:
                losses, latency_ms = estimate_loss(model, train_loader, val_loader, cfg.eval_iters)
                print(f"[{method_name.upper()} | Seed {seed}] step {iter}: val loss {losses['val']:.4f} | latency {latency_ms:.2f}ms")
                history['iter'].append(iter)
                history['val_loss'].append(losses['val'])
                history['train_loss'].append(losses['train'])
                final_latency = latency_ms
                gc.collect()

            for _ in range(grad_accum_steps):
                xb, yb = train_loader.next_batch()

                with torch.amp.autocast('cuda'):
                    logits, task_loss = model(xb, yb)

                    total_loss = task_loss
                    if cfg.method in ['dsc', 'standard_moe']:
                        reg_loss_sum = 0.0
                        
                        for block in model.blocks:
                            if hasattr(block.ffn, 'running_losses') and block.ffn.running_losses:
                                losses = block.ffn.running_losses
                                for k, v in losses.items():
                                    reg_loss_sum += v
                                    
                        total_loss = task_loss + reg_loss_sum

                    loss_scaled = total_loss / grad_accum_steps

                scaler.scale(loss_scaled).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    finally:
        del model
        del optimizer
        del scaler
        torch.cuda.empty_cache()

    return history, final_latency

def print_fairness_report(cfg):
    print(f"\n====== FAIRNESS CHECK: {cfg.method.upper()} ======")
    model = GPT(cfg)
    total_params = sum(p.numel() for p in model.parameters())

    active_params = 0
    if cfg.method == 'dense':
        active_params = total_params
    elif cfg.method == 'standard_moe':
        single_moe_layer_total = sum(p.numel() for p in model.blocks[0].ffn.parameters())
        router_params = cfg.d_model * cfg.num_experts
        expert_dim = cfg.expert_d_ffn
        active_experts = cfg.moe_top_k * (2 * cfg.d_model * expert_dim)
        single_moe_layer_active = router_params + active_experts
        non_ffn_params = total_params - (cfg.n_layer * single_moe_layer_total)
        active_params = non_ffn_params + (cfg.n_layer * single_moe_layer_active)
        
    elif cfg.method == 'dsc':
        single_dmc_layer_total = sum(p.numel() for p in model.blocks[0].ffn.parameters())
        static_params = 2 * cfg.d_model * cfg.dmc_base_d_ffn
        router_params = cfg.d_model * cfg.num_bases
        # Router norm + Gamma also count as active but are negligible
        active_bases = cfg.dmc_top_k * (2 * cfg.d_model) 
        
        single_dmc_layer_active = static_params + router_params + active_bases
        non_ffn_params = total_params - (cfg.n_layer * single_dmc_layer_total)
        active_params = non_ffn_params + (cfg.n_layer * single_dmc_layer_active)

    print(f"1. Total Params: {total_params/1e6:.4f} M")
    print(f"2. Active Params: {active_params/1e6:.4f} M")
    del model

# ==========================================
# 6. Main
# ==========================================
if __name__ == "__main__":
    SEEDS = [42, 1337] 

    agg_results = {
        'dsc': {'hist': [], 'lat': []},
        'standard_moe': {'hist': [], 'lat': []},
        'dense': {'hist': [], 'lat': []}
    }

    print("\n--- Pre-Computation Fairness Checks ---")
    for m in ['dense', 'standard_moe', 'dsc']:
        c = Config(method=m)
        c = auto_configure_model(c)
        print_fairness_report(c)

    print("\nStarting Rigorous Evaluation (Highly Optimized)...")

    for seed in SEEDS:
        print(f"\n>>>>>> RUNNING SEED {seed} <<<<<<")
        h, l = run_experiment('dsc', seed)
        agg_results['dsc']['hist'].append(h)
        agg_results['dsc']['lat'].append(l)

        h, l = run_experiment('standard_moe', seed)
        agg_results['standard_moe']['hist'].append(h)
        agg_results['standard_moe']['lat'].append(l)

        h, l = run_experiment('dense', seed)
        agg_results['dense']['hist'].append(h)
        agg_results['dense']['lat'].append(l)

    plt.figure(figsize=(12, 7))
    colors = {'dsc': 'red', 'standard_moe': 'blue', 'dense': 'green'}
    print("\n" + "="*60)
    print(f"{'METHOD':<15} | {'VAL LOSS (Mean±Std)':<25} | {'LATENCY (ms)':<15}")
    print("-" * 60)

    for method in ['dense', 'standard_moe', 'dsc']:
        all_losses = np.array([h['val_loss'] for h in agg_results[method]['hist']])
        mean_loss = np.mean(all_losses, axis=0)
        std_loss = np.std(all_losses, axis=0)
        iters = agg_results[method]['hist'][0]['iter']

        plt.plot(iters, mean_loss, label=f"{method.upper()}", color=colors[method], linewidth=2)
        plt.fill_between(iters, mean_loss - std_loss, mean_loss + std_loss, color=colors[method], alpha=0.15)

        final_mean = mean_loss[-1]
        final_std = std_loss[-1]
        avg_lat = np.mean(agg_results[method]['lat'])
        print(f"{method:<15} | {final_mean:.4f} ± {final_std:.4f}          | {avg_lat:.2f} ms")

    print("="*60 + "\n")
    plt.xlabel("Iterations")
    plt.ylabel("Validation Loss")
    plt.title("Evaluation: Rigorous DSC (Optimized) vs Baselines")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("publication_results_rigorous_final.png")
    print(" > Saved plot to publication_results_rigorous_final.png")