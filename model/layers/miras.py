"""
MIRAS Framework Implementation
Based on: "It's All Connected: A Journey Through Test-Time Memorization,
Attentional Bias, Retention, and Online Optimization" (Google Research, 2025)

4 Core Components:
1. Memory Architecture (Vector, Matrix, MLP, k-layer MLP, Mosaics)
2. Attentional Bias (Dot-Product, L2, L1, Lp, Huber, KL, Robust, Elastic Net)
3. Retention Gate (L2, KL, Elastic Net, Lq, Bregman, f-Divergence, Shannon)
4. Memory Learning Algorithm (GD, Momentum, Implicit, Newton, Non-parametric)

Novel Models: MONETA, YAAD, MEMORA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Literal, Dict, Any

# =============================================================================
# MEMORY ARCHITECTURES (5 types)
# =============================================================================

class VectorMemory(nn.Module):
    """Type 1: Vector-valued memory (RetNet, LRU style)"""
    def __init__(self, d_model: int):
        super().__init__()
        self.memory = nn.Parameter(torch.zeros(d_model))
        self.alpha = nn.Parameter(torch.ones(1) * 0.9)  # Retention
        
    def forward(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # M_t = α * M_{t-1} + v * k^T (outer product collapsed to vector)
        self.memory.data = self.alpha * self.memory.data + (v * k).mean(dim=0)
        return self.memory
    
    def read(self, q: torch.Tensor) -> torch.Tensor:
        return q * self.memory


class MatrixMemory(nn.Module):
    """Type 2: Matrix-valued memory (DeltaNet, Mamba, GLA style)"""
    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.memory = nn.Parameter(torch.zeros(n_heads, self.head_dim, self.head_dim))
        self.alpha = nn.Parameter(torch.ones(n_heads) * 0.9)
        self.beta = nn.Parameter(torch.ones(n_heads) * 0.1)  # Learning rate
        
    def forward(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Delta rule: M_t = α(I - βkk^T)M_{t-1} + βvk^T
        B, H, D = k.shape[0], self.n_heads, self.head_dim
        k = k.view(B, H, D)
        v = v.view(B, H, D)
        
        for h in range(H):
            kk = torch.outer(k[:, h].mean(0), k[:, h].mean(0))
            vk = torch.outer(v[:, h].mean(0), k[:, h].mean(0))
            I = torch.eye(D, device=k.device)
            self.memory.data[h] = self.alpha[h] * (I - self.beta[h] * kk) @ self.memory.data[h] + self.beta[h] * vk
            
        return self.memory
    
    def read(self, q: torch.Tensor) -> torch.Tensor:
        B, D = q.shape
        q = q.view(B, self.n_heads, self.head_dim)
        out = torch.zeros_like(q)
        for h in range(self.n_heads):
            out[:, h] = q[:, h] @ self.memory[h]
        return out.view(B, D)


class MLPMemory(nn.Module):
    """Type 3: 2-layer MLP Memory (TTT-MLP, Titans, Moneta, Yaad, Memora style)"""
    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        self.d_model = d_model
        self.hidden = d_model * expansion
        
        # M(x) = x + LN(W1 * σ(W2 * x))
        self.W1 = nn.Linear(self.hidden, d_model)
        self.W2 = nn.Linear(d_model, self.hidden)
        self.ln = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        
        # Learnable parameters
        self.eta = nn.Parameter(torch.ones(d_model) * 0.01)  # Learning rate
        self.alpha = nn.Parameter(torch.ones(d_model) * 0.9)  # Retention
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ln(self.W1(self.act(self.W2(x))))
    
    def update(self, k: torch.Tensor, v: torch.Tensor, loss_grad: torch.Tensor):
        """Update memory weights based on gradient"""
        with torch.no_grad():
            for param in [self.W1.weight, self.W2.weight]:
                param.data = self.alpha.mean() * param.data - self.eta.mean() * loss_grad.mean()


class DeepMLPMemory(nn.Module):
    """Type 4: k-layer MLP Memory (Titans-LMM style)"""
    def __init__(self, d_model: int, n_layers: int = 4, expansion: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(d_model, d_model * expansion),
                nn.GELU(),
                nn.Linear(d_model * expansion, d_model),
                nn.LayerNorm(d_model)
            ))
        
        # Momentum-based update (Titans style)
        self.eta = nn.Parameter(torch.ones(d_model) * 0.01)
        self.alpha = nn.Parameter(torch.ones(d_model) * 0.9)
        self.theta = nn.Parameter(torch.ones(d_model) * 0.9)  # Momentum
        self.momentum = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return x
    
    def update_with_momentum(self, grad: torch.Tensor):
        """S_t = θ*S_{t-1} - η*∇L, W_t = α*W_{t-1} - S_t"""
        if self.momentum is None:
            self.momentum = -self.eta * grad
        else:
            self.momentum = self.theta * self.momentum - self.eta * grad


class MemoryMosaics(nn.Module):
    """Type 5: Memory Mosaics - Multiple memory types combined"""
    def __init__(self, d_model: int):
        super().__init__()
        self.vector_mem = VectorMemory(d_model)
        self.matrix_mem = MatrixMemory(d_model)
        self.mlp_mem = MLPMemory(d_model)
        
        self.gate = nn.Linear(d_model, 3)
        
    def forward(self, x: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        gates = F.softmax(self.gate(x), dim=-1)
        
        v_out = self.vector_mem.read(x)
        m_out = self.matrix_mem.read(x)
        mlp_out = self.mlp_mem(x)
        
        return gates[:, 0:1] * v_out + gates[:, 1:2] * m_out + gates[:, 2:3] * mlp_out


# =============================================================================
# ATTENTIONAL BIAS (8 types)
# =============================================================================

class AttentionalBias(nn.Module):
    """Base class for attentional bias objectives"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DotProductBias(AttentionalBias):
    """Type 1: Dot-Product Similarity (Hebbian rule)"""
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return -2 * (pred * target).sum(dim=-1).mean()


class L2RegressionBias(AttentionalBias):
    """Type 2: L2 Regression Loss (Delta rule) - MSE"""
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)


class L1Bias(AttentionalBias):
    """Type 3: L1 Loss (Value-less memory)
    Results in memory with only -1 and +1 values
    """
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)
    
    def gradient(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # ∇ = Sign(Wk - v) * k^T
        return torch.sign(pred - target)


class LpBias(AttentionalBias):
    """Type 4: Lp Loss (p-norm objective)
    Paper shows p=3 is optimal for many tasks
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.eps = eps
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        # Smooth approximation: |x| = sqrt(x^2 + eps)
        abs_diff = torch.sqrt(diff ** 2 + self.eps)
        return (abs_diff ** self.p).mean()
    
    def gradient(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        abs_diff = torch.sqrt(diff ** 2 + self.eps)
        # ∇ = p * Sign(diff) ⊙ |diff|^{p-1}
        # Smooth Sign: tanh(αx)
        smooth_sign = torch.tanh(10 * diff)
        return self.p * smooth_sign * (abs_diff ** (self.p - 1))


class HuberBias(AttentionalBias):
    """Type 5: Huber Loss (Robust to outliers) - YAAD style
    Uses L2 for small errors, L1 for large errors (coping mechanism)
    """
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = nn.Parameter(torch.tensor(delta))
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        abs_diff = torch.abs(diff)
        
        # Huber loss per element
        quadratic = 0.5 * diff ** 2
        linear = self.delta * (abs_diff - 0.5 * self.delta)
        
        return torch.where(abs_diff <= self.delta, quadratic, linear).mean()
    
    def gradient(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        abs_diff = torch.abs(diff)
        
        # L2 gradient for small, L1 gradient for large
        l2_grad = diff
        l1_grad = self.delta * torch.sign(diff)
        
        return torch.where(abs_diff <= self.delta, l2_grad, l1_grad)


class KLDivergenceBias(AttentionalBias):
    """Type 6: KL Divergence"""
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Ensure positive values
        pred_prob = F.softmax(pred, dim=-1)
        target_prob = F.softmax(target, dim=-1)
        return F.kl_div(pred_prob.log(), target_prob, reduction='batchmean')


class RobustBias(AttentionalBias):
    """Type 7: Robust Loss (Value shift resistant)
    L = 0.5 ||M(k) - v||^2 + Δ||M(k) - v|| + 0.5Δ^2
    """
    def __init__(self, delta: float = 0.1):
        super().__init__()
        self.delta = nn.Parameter(torch.tensor(delta))
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        l2_norm = torch.norm(diff, dim=-1)
        
        return (0.5 * l2_norm ** 2 + self.delta * l2_norm + 0.5 * self.delta ** 2).mean()


class ElasticNetBias(AttentionalBias):
    """Type 8: Elastic Net Loss (L1 + L2)"""
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))  # Balance L1 vs L2
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = F.l1_loss(pred, target)
        l2 = F.mse_loss(pred, target)
        return self.alpha * l1 + (1 - self.alpha) * l2


# =============================================================================
# RETENTION GATES (8 types)
# =============================================================================

class RetentionGate(nn.Module):
    """Base class for retention/forget gates"""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
    def forward(self, W_current: torch.Tensor, W_prev: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class L2LocalRetention(RetentionGate):
    """Type 1: L2 Local Regularization
    Ret(W, W_{t-1}) = ||W - W_{t-1}||^2_F
    """
    def __init__(self, d_model: int, eta: float = 0.01):
        super().__init__(d_model)
        self.eta = nn.Parameter(torch.tensor(eta))
        
    def forward(self, W_current: torch.Tensor, W_prev: torch.Tensor) -> torch.Tensor:
        return (1 / (2 * self.eta)) * F.mse_loss(W_current, W_prev, reduction='sum')


class L2GlobalRetention(RetentionGate):
    """Type 2: L2 Global Regularization
    G(W) = ||W||^2_2
    """
    def __init__(self, d_model: int, beta: float = 0.01):
        super().__init__(d_model)
        self.beta = nn.Parameter(torch.tensor(beta))
        
    def forward(self, W_current: torch.Tensor, W_prev: torch.Tensor = None) -> torch.Tensor:
        return (1 / self.beta) * (W_current ** 2).sum()


class KLRetention(RetentionGate):
    """Type 3: KL Divergence Retention (MEMORA style)
    W_t = Softmax(α*log(W_{t-1}) - η*∇L)
    """
    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.alpha = nn.Parameter(torch.ones(d_model) * 0.9)
        self.eta = nn.Parameter(torch.ones(d_model) * 0.01)
        
    def forward(self, W_current: torch.Tensor, W_prev: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        # KL-based update
        log_prev = torch.log(W_prev.abs() + 1e-8)
        return F.softmax(self.alpha * log_prev - self.eta * grad, dim=-1)


class ElasticNetRetention(RetentionGate):
    """Type 4: Elastic Net (Hard + Soft forgetting)
    W_t = S_γ(λ*W_{t-1} - ζ*∇L)
    """
    def __init__(self, d_model: int, gamma: float = 0.1):
        super().__init__(d_model)
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.lambda_ = nn.Parameter(torch.ones(d_model) * 0.9)
        self.zeta = nn.Parameter(torch.ones(d_model) * 0.01)
        
    def soft_threshold(self, z: torch.Tensor) -> torch.Tensor:
        """S_γ(z) = sign(z) * max(0, |z| - γ)"""
        return torch.sign(z) * F.relu(torch.abs(z) - self.gamma)
    
    def forward(self, W_prev: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        return self.soft_threshold(self.lambda_ * W_prev - self.zeta * grad)


class LqRetention(RetentionGate):
    """Type 5: Lq Memory Stability
    R(W) = (1/(2(q-1))) * ||W||^2_q
    """
    def __init__(self, d_model: int, q: float = 4.0):
        super().__init__(d_model)
        self.q = q
        self.eta = nn.Parameter(torch.tensor(0.01))
        
    def forward(self, A: torch.Tensor) -> torch.Tensor:
        # W_t = A_t / ||A_t||^{(p-2)/p} where p = q/(q-1)
        p = self.q / (self.q - 1)
        norm = torch.norm(A, p=p, dim=-1, keepdim=True)
        return A / (norm ** ((p - 2) / p) + 1e-8)


class BregmanRetention(RetentionGate):
    """Type 6: Bregman Divergence Retention
    Uses sigmoid for bounded updates
    """
    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.eta = nn.Parameter(torch.tensor(0.01))
        
    def forward(self, W_prev: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        # W_t = σ(ln(W_t/(1-W_t)) - η*∇L)
        # Sigmoid-based Bregman divergence ensures W ∈ (0, 1)
        W_clipped = torch.clamp(W_prev, 1e-6, 1 - 1e-6)
        logit = torch.log(W_clipped / (1 - W_clipped))
        return torch.sigmoid(logit - self.eta * grad)


class FDivergenceRetention(RetentionGate):
    """Type 7: f-Divergence over Probability Simplex
    Constrains W to lie in scaled probability simplex
    """
    def __init__(self, d_model: int, c: float = 1.0):
        super().__init__(d_model)
        self.c = c  # Simplex scale
        self.eta = nn.Parameter(torch.tensor(0.01))
        
    def forward(self, W_prev: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        # Project onto scaled simplex
        W_new = W_prev - self.eta * grad
        W_new = F.relu(W_new)  # Ensure non-negative
        W_new = self.c * W_new / (W_new.sum(dim=-1, keepdim=True) + 1e-8)
        return W_new


class ShannonEntropyRetention(RetentionGate):
    """Type 8: Shannon Entropy Retention
    G(W) = Σ W_jl * log(W_jl)
    """
    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, W: torch.Tensor) -> torch.Tensor:
        W_prob = F.softmax(W, dim=-1)
        entropy = -(W_prob * torch.log(W_prob + 1e-8)).sum()
        return self.alpha * entropy


# =============================================================================
# MEMORY LEARNING ALGORITHMS (6 types)
# =============================================================================

class MemoryOptimizer:
    """Base class for memory learning algorithms"""
    def step(self, memory: nn.Module, grad: torch.Tensor):
        raise NotImplementedError


class GradientDescent(MemoryOptimizer):
    """Type 1: Standard Gradient Descent"""
    def __init__(self, lr: float = 0.01):
        self.lr = lr
        
    def step(self, param: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        return param - self.lr * grad


class GDWithMomentum(MemoryOptimizer):
    """Type 2: Gradient Descent with Momentum (Titans style)"""
    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = None
        
    def step(self, param: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        if self.velocity is None:
            self.velocity = torch.zeros_like(grad)
        self.velocity = self.momentum * self.velocity - self.lr * grad
        return param + self.velocity


class ImplicitGD(MemoryOptimizer):
    """Type 3: Implicit Gradient Descent (Longhorn style)
    M_t = (I - βkk^T/(1+βk^Tk))M_{t-1} + β/(1+k^Tk*β) * x * k^T
    """
    def __init__(self, beta: float = 0.1):
        self.beta = beta
        
    def step(self, M: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        kk = torch.outer(k.mean(0), k.mean(0))
        ktk = (k * k).sum()
        
        I = torch.eye(M.shape[0], device=M.device)
        factor1 = I - (self.beta * kk) / (1 + self.beta * ktk)
        factor2 = self.beta / (1 + ktk * self.beta)
        
        return factor1 @ M + factor2 * torch.outer(v.mean(0), k.mean(0))


class NewtonMethod(MemoryOptimizer):
    """Type 4: Newton's Method (Mesa-layer style)
    Uses second-order information for faster convergence
    """
    def __init__(self, lr: float = 0.1, damping: float = 1e-4):
        self.lr = lr
        self.damping = damping
        
    def step(self, param: torch.Tensor, grad: torch.Tensor, hessian: torch.Tensor = None) -> torch.Tensor:
        if hessian is None:
            # Approximate with identity (falls back to GD)
            return param - self.lr * grad
        
        # Newton step: W_new = W - H^{-1} * grad
        H_damped = hessian + self.damping * torch.eye(hessian.shape[0], device=hessian.device)
        try:
            H_inv = torch.linalg.inv(H_damped)
            return param - self.lr * (H_inv @ grad)
        except:
            return param - self.lr * grad


class NonParametricSolution(MemoryOptimizer):
    """Type 5: Non-parametric Solutions (Transformer/Attention style)
    Uses Nadaraya-Watson estimator
    """
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        
    def solve(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # Softmax attention as non-parametric solution
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.temperature * math.sqrt(Q.shape[-1]))
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, V)


class MultiStepGD(MemoryOptimizer):
    """Type 6: Multiple GD steps per token (DeltaProduct style)"""
    def __init__(self, lr: float = 0.01, n_steps: int = 3):
        self.lr = lr
        self.n_steps = n_steps
        
    def step(self, param: torch.Tensor, grad_fn, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n_steps):
            grad = grad_fn(param, k, v)
            param = param - self.lr * grad
        return param


# =============================================================================
# NOVEL MODELS: MONETA, YAAD, MEMORA
# =============================================================================

class Moneta(nn.Module):
    """MONETA: Lp attentional bias + Lq retention
    
    (p,q)-Moneta uses:
    - 2-layer MLP memory with GELU + residual + LayerNorm
    - Lp attentional bias (p=3 optimal)
    - Lq retention gate (hybrid with L2)
    - Gradient descent optimizer
    
    Update: A_t = α*A_{t-1} - η*∇ℓ_p, W_t = A_t/||A_t||^{(q-2)/q}
    """
    def __init__(self, d_model: int, p: float = 3.0, q: float = 4.0, expansion: int = 4):
        super().__init__()
        self.d_model = d_model
        self.p = p
        self.q = q
        
        # Memory architecture: 2-layer MLP
        self.memory = MLPMemory(d_model, expansion)
        
        # Attentional bias: Lp loss
        self.bias = LpBias(p=p)
        
        # Retention: Lq + L2 hybrid
        self.lq_retention = LqRetention(d_model, q=q)
        self.l2_retention = L2GlobalRetention(d_model)
        
        # Learnable parameters (channel-wise)
        self.alpha = nn.Parameter(torch.ones(d_model) * 0.9)
        self.eta = nn.Parameter(torch.ones(d_model) * 0.01)
        
        # Accumulator for Lq normalization
        self.register_buffer('A', torch.zeros(d_model))
        
    def forward(self, k: torch.Tensor, v: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Compute prediction
        pred = self.memory(k)
        
        # Compute Lp loss
        loss = self.bias(pred, v)
        
        # Compute gradient
        grad = self.bias.gradient(pred, v)
        
        # Update accumulator: A_t = α*A_{t-1} - η*∇ℓ_p
        self.A = self.alpha * self.A - self.eta * grad.mean(0)
        
        # Apply Lq retention: W_t = A_t / ||A_t||^{(q-2)/q}
        W_normalized = self.lq_retention(self.A)
        
        return {
            'output': pred,
            'loss': loss,
            'memory_state': W_normalized
        }


class Yaad(nn.Module):
    """YAAD: Huber loss + L2 retention (coping mechanism)
    
    Uses:
    - 2-layer MLP memory
    - Huber attentional bias (robust to outliers)
    - L2 local + global retention
    - Gradient descent optimizer
    
    Update: W_t = α*W_{t-1} - η*∇L (L2 if small, L1 if large)
    """
    def __init__(self, d_model: int, delta: float = 1.0, expansion: int = 4):
        super().__init__()
        self.d_model = d_model
        
        # Memory architecture
        self.memory = MLPMemory(d_model, expansion)
        
        # Attentional bias: Huber loss
        self.bias = HuberBias(delta=delta)
        
        # Retention: L2 local + global
        self.local_retention = L2LocalRetention(d_model)
        self.global_retention = L2GlobalRetention(d_model)
        
        # Learnable parameters (channel-wise)
        self.alpha = nn.Parameter(torch.ones(d_model) * 0.9)
        self.eta = nn.Parameter(torch.ones(d_model) * 0.01)
        self.delta = nn.Parameter(torch.ones(d_model) * delta)
        
    def forward(self, k: torch.Tensor, v: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred = self.memory(k)
        loss = self.bias(pred, v)
        grad = self.bias.gradient(pred, v)
        
        # Huber-based update
        return {
            'output': pred,
            'loss': loss,
            'grad': grad
        }


class Memora(nn.Module):
    """MEMORA: L2 loss + KL divergence retention
    
    Uses:
    - 2-layer MLP memory
    - L2 attentional bias
    - KL divergence retention
    - Gradient descent optimizer
    
    Update: W_t = Softmax(α*log(W_{t-1}) - η*∇ℓ_2)
    """
    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        self.d_model = d_model
        
        # Memory architecture
        self.memory = MLPMemory(d_model, expansion)
        
        # Attentional bias: L2
        self.bias = L2RegressionBias()
        
        # Retention: KL divergence
        self.retention = KLRetention(d_model)
        
        # Learnable parameters
        self.alpha = nn.Parameter(torch.ones(d_model) * 0.9)
        self.eta = nn.Parameter(torch.ones(d_model) * 0.01)
        
        # Previous state
        self.register_buffer('W_prev', torch.ones(d_model) / d_model)
        
    def forward(self, k: torch.Tensor, v: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred = self.memory(k)
        loss = self.bias(pred, v)
        
        # Compute gradient
        grad = 2 * (pred - v)  # L2 gradient
        
        # KL-based update: W_t = Softmax(α*log(W_{t-1}) - η*∇L)
        log_prev = torch.log(self.W_prev.abs() + 1e-8)
        W_new = F.softmax(self.alpha * log_prev - self.eta * grad.mean(0), dim=-1)
        self.W_prev = W_new
        
        return {
            'output': pred,
            'loss': loss,
            'memory_state': W_new
        }


# =============================================================================
# ARCHITECTURAL FEATURES
# =============================================================================

class DepthwiseSeparableConv1d(nn.Module):
    """1D Depthwise-separable convolution (kernel=4)"""
    def __init__(self, d_model: int, kernel_size: int = 4):
        super().__init__()
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size, 
                                    padding=kernel_size-1, groups=d_model)
        self.pointwise = nn.Conv1d(d_model, d_model, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D) -> (B, D, L)
        x = x.transpose(1, 2)
        x = self.depthwise(x)[:, :, :-self.depthwise.kernel_size[0]+1]
        x = self.pointwise(x)
        return x.transpose(1, 2)


class SwiGLU(nn.Module):
    """SwiGLU activation"""
    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        hidden = d_model * expansion
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class RoPE(nn.Module):
    """Rotary Position Encoding"""
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        self.d_model = d_model
        
        # Precompute frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute position encodings
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer('cos', freqs.cos())
        self.register_buffer('sin', freqs.sin())
        
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        # x shape: (B, L, H, D) or (B, L, D)
        if x.dim() == 4:
            B, L, H, D = x.shape
        else:
            B, L, D = x.shape
            H = 1
            x = x.unsqueeze(2)
        
        seq_len = L
        cos = self.cos[offset:offset+seq_len].unsqueeze(0).unsqueeze(2)  # (1, L, 1, D/2)
        sin = self.sin[offset:offset+seq_len].unsqueeze(0).unsqueeze(2)
        
        # Rotate
        x1 = x[..., ::2]  # (B, L, H, D/2)
        x2 = x[..., 1::2]
        
        out = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        
        if H == 1:
            out = out.squeeze(2)
        return out


class GatedOutput(nn.Module):
    """Gated output layer"""
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return torch.sigmoid(self.gate(x)) * self.proj(x)


class LowRankProjection(nn.Module):
    """Low-rank projection for channel-wise parameters (k=32 or 64)"""
    def __init__(self, d_model: int, rank: int = 32):
        super().__init__()
        self.down = nn.Linear(d_model, rank)
        self.up = nn.Linear(rank, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class L2Normalize(nn.Module):
    """L2 normalization for q, k stability"""
    def __init__(self, dim: int = -1, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)


# =============================================================================
# MIRAS LAYER (Complete Implementation)
# =============================================================================

class MIRASLayer(nn.Module):
    """
    Complete MIRAS Layer with all features:
    - Configurable memory architecture
    - Configurable attentional bias
    - Configurable retention gate
    - Configurable optimizer
    - All architectural features (Conv, RoPE, SwiGLU, etc.)
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        memory_type: Literal['vector', 'matrix', 'mlp', 'deep_mlp', 'mosaics'] = 'mlp',
        bias_type: Literal['dot', 'l2', 'l1', 'lp', 'huber', 'kl', 'robust', 'elastic'] = 'l2',
        retention_type: Literal['l2_local', 'l2_global', 'kl', 'elastic', 'lq', 'bregman', 'fdiv', 'shannon'] = 'l2_local',
        optimizer_type: Literal['gd', 'momentum', 'implicit', 'newton', 'nonparam', 'multistep'] = 'gd',
        use_conv: bool = True,
        use_rope: bool = True,
        use_l2_norm: bool = True,
        expansion: int = 4,
        rank: int = 32,
        p: float = 3.0,  # For Lp bias
        q: float = 4.0,  # For Lq retention
        delta: float = 1.0,  # For Huber
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # QKV projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Optional 1D Conv after projections
        self.use_conv = use_conv
        if use_conv:
            self.q_conv = DepthwiseSeparableConv1d(d_model)
            self.k_conv = DepthwiseSeparableConv1d(d_model)
            self.v_conv = DepthwiseSeparableConv1d(d_model)
        
        # Optional RoPE
        self.use_rope = use_rope
        if use_rope:
            self.rope = RoPE(self.head_dim)
            
        # Optional L2 normalization
        self.use_l2_norm = use_l2_norm
        if use_l2_norm:
            self.l2_norm = L2Normalize()
            
        # Low-rank projections for channel-wise parameters
        self.eta_proj = LowRankProjection(d_model, rank)
        self.alpha_proj = LowRankProjection(d_model, rank)
        self.delta_proj = LowRankProjection(d_model, rank)
        
        # Memory architecture
        self.memory_type = memory_type
        if memory_type == 'vector':
            self.memory = VectorMemory(d_model)
        elif memory_type == 'matrix':
            self.memory = MatrixMemory(d_model, n_heads)
        elif memory_type == 'mlp':
            self.memory = MLPMemory(d_model, expansion)
        elif memory_type == 'deep_mlp':
            self.memory = DeepMLPMemory(d_model, n_layers=4, expansion=expansion)
        elif memory_type == 'mosaics':
            self.memory = MemoryMosaics(d_model)
            
        # Attentional bias
        self.bias_type = bias_type
        if bias_type == 'dot':
            self.bias = DotProductBias()
        elif bias_type == 'l2':
            self.bias = L2RegressionBias()
        elif bias_type == 'l1':
            self.bias = L1Bias()
        elif bias_type == 'lp':
            self.bias = LpBias(p=p)
        elif bias_type == 'huber':
            self.bias = HuberBias(delta=delta)
        elif bias_type == 'kl':
            self.bias = KLDivergenceBias()
        elif bias_type == 'robust':
            self.bias = RobustBias(delta=delta)
        elif bias_type == 'elastic':
            self.bias = ElasticNetBias()
            
        # Retention gate
        self.retention_type = retention_type
        if retention_type == 'l2_local':
            self.retention = L2LocalRetention(d_model)
        elif retention_type == 'l2_global':
            self.retention = L2GlobalRetention(d_model)
        elif retention_type == 'kl':
            self.retention = KLRetention(d_model)
        elif retention_type == 'elastic':
            self.retention = ElasticNetRetention(d_model)
        elif retention_type == 'lq':
            self.retention = LqRetention(d_model, q=q)
        elif retention_type == 'bregman':
            self.retention = BregmanRetention(d_model)
        elif retention_type == 'fdiv':
            self.retention = FDivergenceRetention(d_model)
        elif retention_type == 'shannon':
            self.retention = ShannonEntropyRetention(d_model)
            
        # Output
        self.output_gate = GatedOutput(d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # RMSNorm
        self.norm = nn.RMSNorm(d_model) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, L, D = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Apply optional conv
        if self.use_conv:
            q = self.q_conv(q)
            k = self.k_conv(k)
            v = self.v_conv(v)
            
        # Apply optional L2 norm
        if self.use_l2_norm:
            q = self.l2_norm(q)
            k = self.l2_norm(k)
            
        # Apply optional RoPE
        if self.use_rope:
            q = q.view(B, L, self.n_heads, self.head_dim)
            k = k.view(B, L, self.n_heads, self.head_dim)
            q = self.rope(q).view(B, L, D)
            k = self.rope(k).view(B, L, D)
        
        # Memory forward pass
        if self.memory_type == 'mosaics':
            out = self.memory(x, k, v)
        else:
            out = self.memory(k)
            
        # Compute attentional bias loss
        loss = self.bias(out, v)
        
        # Apply output gate
        out = self.output_gate(out)
        out = self.out_proj(out)
        
        # Residual + Norm
        out = self.norm(x + out)
        
        return {
            'output': out,
            'loss': loss
        }


# =============================================================================
# MIRAS BLOCK (Full Transformer-like block)
# =============================================================================

class MIRASBlock(nn.Module):
    """
    Full MIRAS Block (replaces Transformer block):
    - MIRAS Layer (memory + attention replacement)
    - SwiGLU MLP
    - Residual connections
    - RMSNorm
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        expansion: int = 4,
        memory_type: str = 'mlp',
        bias_type: str = 'l2',
        retention_type: str = 'l2_local',
        **kwargs
    ):
        super().__init__()
        
        self.norm1 = nn.RMSNorm(d_model) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model) if hasattr(nn, 'RMSNorm') else nn.LayerNorm(d_model)
        
        self.miras = MIRASLayer(
            d_model=d_model,
            n_heads=n_heads,
            memory_type=memory_type,
            bias_type=bias_type,
            retention_type=retention_type,
            **kwargs
        )
        
        self.mlp = SwiGLU(d_model, expansion)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # MIRAS attention replacement
        h = self.norm1(x)
        miras_out = self.miras(h)
        x = x + miras_out['output']
        
        # SwiGLU MLP
        h = self.norm2(x)
        x = x + self.mlp(h)
        
        return {
            'output': x,
            'miras_loss': miras_out['loss']
        }


# =============================================================================
# UTILITY: Get all MIRAS configurations
# =============================================================================

def get_miras_config(model_name: str = 'moneta') -> Dict[str, Any]:
    """Get predefined MIRAS configurations"""
    configs = {
        'moneta': {
            'memory_type': 'mlp',
            'bias_type': 'lp',
            'retention_type': 'lq',
            'p': 3.0,
            'q': 4.0
        },
        'yaad': {
            'memory_type': 'mlp',
            'bias_type': 'huber',
            'retention_type': 'l2_local',
            'delta': 1.0
        },
        'memora': {
            'memory_type': 'mlp',
            'bias_type': 'l2',
            'retention_type': 'kl'
        },
        'titans': {
            'memory_type': 'deep_mlp',
            'bias_type': 'l2',
            'retention_type': 'l2_global',
            'optimizer_type': 'momentum'
        },
        'deltanet': {
            'memory_type': 'matrix',
            'bias_type': 'l2',
            'retention_type': 'l2_local'
        },
        'retnet': {
            'memory_type': 'vector',
            'bias_type': 'dot',
            'retention_type': 'l2_local'
        }
    }
    return configs.get(model_name, configs['moneta'])




class SlidingWindowAttention(nn.Module):
    """Sliding Window Attention (SWA) for Hybrid models
    
    Used in combination with MIRAS layers for long sequences.
    Window size determines local attention span.
    """
    def __init__(self, d_model: int, n_heads: int = 8, window_size: int = 256):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Create sliding window mask
        mask = torch.ones(L, L, device=x.device, dtype=torch.bool)
        for i in range(L):
            start = max(0, i - self.window_size // 2)
            end = min(L, i + self.window_size // 2 + 1)
            mask[i, start:end] = False
        
        # Attention with mask
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        
        return self.out_proj(out)


class HybridMIRASBlock(nn.Module):
    """Hybrid MIRAS + Sliding Window Attention Block
    
    Combines MIRAS layer with SWA for best of both:
    - MIRAS: Memory-based learning with retention
    - SWA: Local attention for fine-grained patterns
    """
    def __init__(self, d_model: int, n_heads: int = 8, window_size: int = 256, 
                 miras_config: str = 'yaad', expansion: int = 4):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # MIRAS layer
        config = get_miras_config(miras_config)
        config = {k: v for k, v in config.items() if k != 'optimizer_type'}
        self.miras = MIRASLayer(d_model=d_model, n_heads=n_heads, **config)
        
        # Sliding Window Attention
        self.swa = SlidingWindowAttention(d_model, n_heads, window_size)
        
        # MLP
        self.mlp = SwiGLU(d_model, expansion)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # MIRAS
        h = self.norm1(x)
        miras_out = self.miras(h)
        x = x + miras_out['output']
        
        # SWA
        h = self.norm2(x)
        x = x + self.swa(h)
        
        # MLP
        h = self.norm3(x)
        x = x + self.mlp(h)
        
        return {
            'output': x,
            'miras_loss': miras_out.get('loss', torch.tensor(0.0))
        }



def list_all_features() -> Dict[str, list]:
    """List all available MIRAS features"""
    return {
        'memory_architectures': ['vector', 'matrix', 'mlp', 'deep_mlp', 'mosaics'],
        'attentional_biases': ['dot', 'l2', 'l1', 'lp', 'huber', 'kl', 'robust', 'elastic'],
        'retention_gates': ['l2_local', 'l2_global', 'kl', 'elastic', 'lq', 'bregman', 'fdiv', 'shannon'],
        'optimizers': ['gd', 'momentum', 'implicit', 'newton', 'nonparam', 'multistep'],
        'architectural_features': [
            'depthwise_conv', 'rope', 'l2_norm', 'swiglu', 'rmsnorm', 
            'gated_output', 'low_rank_proj', 'residual', 'channel_wise_params', 'hybrid_swa'
        ],
        'novel_models': ['moneta', 'yaad', 'memora']
    }
