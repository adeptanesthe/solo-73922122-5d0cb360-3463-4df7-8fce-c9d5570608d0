"""固定权重的小型因果 Transformer（CPU、无外部下载）。

两个模型：
  - 目标模型：4 层、4 头、d_model=128
  - 草稿模型：1 层、2 头、d_model=128（更小）
二者共用同一套 token 嵌入与输出投影（LM head），词表也共用。

支持两种前向：
  - 带 KV 缓存的增量前向（推测解码实际使用，不重算已缓存前缀）
  - 无缓存的整序列前向（贪心基线逐 token 调用，每轮重算全部前缀）
两条路径走同一份注意力代码，仅 KV 缓存与遮罩不同，保证数值逐位一致。

并发模型（关键）：
  - 权重（参数、嵌入、LM head）是进程内唯一的固定单例，所有请求共享、只读；
  - KV 缓存与前向计数是**每请求独立**的可变状态，由调用方创建
    :class:`ModelState` 并在每次前向时显式传入，模型对象本身不再保存
    任何缓存/计数，因此多个请求（线程）并发推理时互不串扰：
    一个请求在推理中途停顿，另一个请求跑完只会改动它自己的状态。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import tokenizer

# 固定随机种子，保证每次构建的权重完全一致（无需下载模型文件）
_WEIGHT_SEED = 20261116
# 草稿模型在共享输出层之外的受控扰动种子与尺度。
# 未训练模型若完全独立初始化，argmax 几乎必然不同（接近 100% 拒绝）；
# 完全一致又不会发生拒绝。该尺度经实测使首 token 翻转率约 30%，
# 从而同一服务内可同时观测到整块接受与拒绝两种路径。
_DRAFT_NOISE_SEED = 99
_DRAFT_NOISE_SIGMA = 0.125
VOCAB_SIZE = tokenizer.VOCAB_SIZE
D_MODEL = 128
MAX_POS = 256


class KVCache:
    """单个请求在单个模型上的 KV 缓存。

    每层保存一个 (k, v)，形状均为 (1, n_heads, seq_len, head_dim)。
    缓存对象随请求创建、随请求释放，绝不挂在共享模型上。
    """

    def __init__(self, n_layers: int, n_heads: int, head_dim: int):
        self._layers: list[tuple[torch.Tensor, torch.Tensor]] = [
            (torch.empty(1, n_heads, 0, head_dim),
             torch.empty(1, n_heads, 0, head_dim))
            for _ in range(n_layers)
        ]
        self.cache_len = 0

    def truncate(self, keep_len: int) -> None:
        """把 KV 缓存截断到前 keep_len 个位置（拒绝后回滚用）。

        keep_len 不小于当前缓存长度时不做任何操作（调用方随后会增量补齐）。
        """
        if keep_len >= self.cache_len:
            return
        self._layers = [(k[:, :, :keep_len, :].contiguous(),
                         v[:, :, :keep_len, :].contiguous())
                        for k, v in self._layers]
        self.cache_len = keep_len

    def __len__(self) -> int:
        return self.cache_len


class ModelState:
    """单次请求在单个模型上的全部可变状态：KV 缓存 + 前向计数器。

    每次推理请求各自持有一个实例；模型权重在请求间共享且只读，
    本对象上的缓存与计数互不影响，从而支持多线程并发对照。
    """

    def __init__(self, model: "CausalTransformer"):
        self.cache = KVCache(model.n_layers, model.n_heads, model.head_dim)
        self.forward_count = 0


class CausalTransformer(nn.Module):
    """带 KV 缓存的因果 Transformer。

    权重在进程内只构建一次并被所有请求共享；前向所需的可变状态
    （KV 缓存、前向计数）通过 ``state`` 参数显式传入，模型实例本身无状态，
    可被任意线程并发调用。
    """

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.head_dim = d_model // n_heads

        # 嵌入与输出投影在两个模型间共享（构建时传入同一模块）
        self.tok_emb: nn.Embedding | None = None
        self.pos_emb = nn.Embedding(MAX_POS, d_model)
        self.blocks = nn.ModuleList(
            [nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head: nn.Linear | None = None

    # ---- 前向 ----
    def _attention_layer(self, x: torch.Tensor,
                         layer: nn.TransformerEncoderLayer,
                         layer_idx: int, use_cache: bool,
                         state: ModelState | None) -> torch.Tensor:
        B, T, _ = x.shape
        # norm_first 结构，缓存/无缓存两条路径完全共用这份计算
        h = layer.norm1(x)
        w = layer.self_attn.in_proj_weight
        q_b, k_b, v_b = layer.self_attn.in_proj_bias.chunk(3)
        q = F.linear(h, w[:self.d_model], q_b)
        k = F.linear(h, w[self.d_model:2 * self.d_model], k_b)
        v = F.linear(h, w[2 * self.d_model:], v_b)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        if use_cache:
            assert state is not None, "use_cache=True 必须传入每请求的 ModelState"
            pk, pv = state.cache._layers[layer_idx]
            past_len = pk.shape[2]
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
            state.cache._layers[layer_idx] = (k, v)
            if T == 1:
                # 单 token 增量：历史 KV 全部可见，无需遮挡
                attn_mask = None
            else:
                # 多 token 增量：历史部分全可见，新 token 之间走因果遮挡
                left = torch.zeros(T, past_len)
                right = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
                attn_mask = torch.cat([left, right], dim=1)
        else:
            # 无缓存整序列：完整因果三角遮挡
            attn_mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = layer.self_attn.out_proj(out)
        x = x + layer.dropout1(out)
        h = layer.norm2(x)
        h = layer.linear2(layer.dropout(F.gelu(layer.linear1(h))))
        x = x + layer.dropout2(h)
        return x

    def forward(self, token_ids: list[int] | torch.Tensor,
                use_cache: bool,
                state: ModelState | None = None) -> torch.Tensor:
        """前向计算，返回 logits。

        - 传入 token id 列表（或 1D 张量）：返回形状 (seq_len, vocab_size)；
        - 传入 2D 张量：返回 (batch, seq_len, vocab_size)，仅无缓存模式支持。
        use_cache=True 时走增量路径并追加/复用 ``state`` 中的 KV 缓存；
        use_cache=False 时整序列重算（基线路径），不触碰任何缓存。
        每次调用使该请求的 ``state.forward_count`` 加 1（无缓存调用若未
        传入 state 则不计数，仅用于独立的底层校验）。
        """
        batched = isinstance(token_ids, torch.Tensor) and token_ids.dim() == 2
        if isinstance(token_ids, list):
            x_in = torch.tensor([token_ids], dtype=torch.long)
        elif token_ids.dim() == 1:
            x_in = token_ids.view(1, -1)
        else:
            x_in = token_ids
        T = x_in.shape[1]

        if use_cache:
            if state is None:
                raise ValueError("use_cache=True 时必须传入该请求专属的 ModelState")
            if batched:
                raise ValueError("带 KV 缓存的增量前向不支持 batch 输入")
            state.forward_count += 1
            positions = torch.arange(state.cache.cache_len,
                                     state.cache.cache_len + T)
        else:
            if state is not None:
                state.forward_count += 1
            positions = torch.arange(T)

        h = self.tok_emb(x_in) + self.pos_emb(positions)
        for i, layer in enumerate(self.blocks):
            h = self._attention_layer(h, layer, i, use_cache, state)
        logits = self.lm_head(self.ln_f(h))
        if use_cache:
            state.cache.cache_len += T
        return logits if batched else logits.squeeze(0)


def _build_pair() -> tuple[CausalTransformer, CausalTransformer]:
    """以固定种子构建草稿/目标模型，共享嵌入与 LM head。

    草稿模型的 Transformer 主体在同一确定性 RNG 流初始化后，再用独立的
    固定种子生成器注入受控高斯扰动，使两模型既有分歧（触发拒绝）又不会
    完全不同（保留整块接受）。全部权重确定可复现，无任何下载。
    """
    torch.manual_seed(_WEIGHT_SEED)
    shared_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
    shared_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
    # 权重绑定：head 权重与嵌入转置共享，保证两个模型输出层完全一致
    shared_head.weight = shared_emb.weight

    target = CausalTransformer(VOCAB_SIZE, D_MODEL, n_heads=4, n_layers=4)
    draft = CausalTransformer(VOCAB_SIZE, D_MODEL, n_heads=2, n_layers=1)
    target.tok_emb = draft.tok_emb = shared_emb
    target.lm_head = draft.lm_head = shared_head

    # 对草稿主体注入固定种子的受控扰动（共享的嵌入/输出层不扰动）
    noise_gen = torch.Generator().manual_seed(_DRAFT_NOISE_SEED)
    with torch.no_grad():
        for p in draft.blocks.parameters():
            p.add_(_DRAFT_NOISE_SIGMA * torch.randn(p.shape, generator=noise_gen))
        for p in draft.pos_emb.parameters():
            p.add_(_DRAFT_NOISE_SIGMA * torch.randn(p.shape, generator=noise_gen))
        for p in draft.ln_f.parameters():
            p.add_(0.1 * _DRAFT_NOISE_SIGMA
                   * torch.randn(p.shape, generator=noise_gen))

    # 所有参数均确定可复现
    target.eval()
    draft.eval()
    for p in target.parameters():
        p.requires_grad_(False)
    for p in draft.parameters():
        p.requires_grad_(False)
    return draft, target


_DRAFT: CausalTransformer | None = None
_TARGET: CausalTransformer | None = None


def get_models() -> tuple[CausalTransformer, CausalTransformer]:
    """返回进程内单例 (draft, target)。

    仅权重是单例且只读；每个请求必须另行创建自己的 :class:`ModelState`，
    不得在模型对象上共享 KV 缓存或前向计数。
    """
    global _DRAFT, _TARGET
    if _TARGET is None:
        _DRAFT, _TARGET = _build_pair()
    return _DRAFT, _TARGET


def greedy_argmax(logits: torch.Tensor) -> int:
    """贪心取 argmax；平局取词表序号小者（torch.argmax 天然返回首个最大值）。"""
    return int(torch.argmax(logits, dim=-1).item())
