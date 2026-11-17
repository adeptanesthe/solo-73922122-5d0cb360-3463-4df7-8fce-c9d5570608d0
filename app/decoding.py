"""两种解码方式：无缓存逐 token 贪心基线 vs 带 KV 缓存的推测解码。

正确性约定：推测解码在任何情况下（拒绝、整块接受、EOS 截断、达到上限）
输出的 token 序列都必须与贪心基线完全一致；argmax 平局时取词表序号小者。

并发约定：草稿/目标模型的权重是进程内共享的固定单例（只读），而 KV 缓存
与前向计数是**每次调用独立创建的** :class:`~app.model.ModelState`，随调用
栈创建与释放。因此多个请求可以在不同线程中并发调用本模块的函数，互不
串扰：一个请求在推理中途停顿，另一个请求完成只会写它自己的状态。

缓存不变量（每轮开始时）：
  - 目标模型：缓存覆盖到确认前缀的倒数第二个 token，另用 pending 保存
    上一轮留下的 tail（纠正/补词 token）。本轮目标前向输入
    ``[pending] + 草稿块``（首轮为完整提示词+草稿块），一次前向后缓存恰好
    覆盖到 pending；被拒绝的草稿位置通过缓存截断丢弃，绝不重算已缓存前缀。
    因此目标前向次数 == 推测轮数。
  - 草稿模型：KV 缓存对齐到完整确认前缀，其尾位置输出的 logits 即对下一
    token 的预测，首个草稿 token 直接取它，无需重复喂尾 token。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from . import tokenizer
from .model import CausalTransformer, ModelState, greedy_argmax


@dataclass
class RoundTrace:
    """推测解码单轮的可观测信息（用于网页对照）。"""
    index: int
    draft_tokens: list[int]
    target_predictions: list[int]
    accepted: int
    decision: str                    # "reject" | "all_accepted"
    target_tokens: list[int]         # 该轮确认的新 token（接受+纠正/补词）
    target_forward_input_len: int    # 该轮目标前向一次吃掉的输入长度


@dataclass
class DecodeResult:
    prompt_ids: list[int]
    new_tokens: list[int]
    target_forward_count: int
    draft_forward_count: int
    rounds: list[RoundTrace] = field(default_factory=list)
    stopped_by_eos: bool = False
    # 本次调用私有的每模型状态（独立 KV 缓存与计数）。并发场景下各请求
    # 持有不同实例；测试可在调用结束后检查其中的缓存长度不变量。
    target_state: ModelState | None = None
    draft_state: ModelState | None = None


def greedy_baseline(target: CausalTransformer, prompt_ids: list[int],
                    max_new: int) -> DecodeResult:
    """目标模型无缓存逐 token 贪心生成。

    每个新 token 都对"完整前缀"做一次全新前向（无 KV 缓存），
    因此目标前向次数 == 实际生成的新 token 数（含 EOS）。
    """
    target_state = ModelState(target)
    seq = list(prompt_ids)
    new_tokens: list[int] = []
    stopped = False
    for _ in range(max_new):
        logits = target.forward(seq, use_cache=False, state=target_state)
        nxt = greedy_argmax(logits[-1])
        seq.append(nxt)
        new_tokens.append(nxt)
        if nxt == tokenizer.EOS_ID:
            stopped = True
            break
    return DecodeResult(
        prompt_ids=list(prompt_ids),
        new_tokens=new_tokens,
        target_forward_count=target_state.forward_count,
        draft_forward_count=0,
        stopped_by_eos=stopped,
        target_state=target_state,
    )


def speculative_decode(draft: CausalTransformer, target: CausalTransformer,
                       prompt_ids: list[int], k: int,
                       max_new: int) -> DecodeResult:
    """带 KV 缓存的推测解码。

    草稿与目标各自使用本调用私有的 :class:`ModelState`（独立 KV 缓存与
    前向计数器），与其他并发请求完全隔离。
    """
    assert 1 <= k <= 4
    target_state = ModelState(target)
    draft_state = ModelState(draft)

    seq = list(prompt_ids)          # 已确认序列（含提示词）
    new_tokens: list[int] = []
    traces: list[RoundTrace] = []
    stopped = False
    # pending：上一轮确认但尚未写入目标 KV 的 tail（首轮为 None）
    pending: int | None = None

    # 草稿预热提示词前缀（1 次前向），拿到首个草稿 token 的预测 logits。
    # 目标模型不单独预热：首轮把"提示词+草稿块"合并为一次前向。
    draft_next_logits: torch.Tensor = draft.forward(
        prompt_ids, use_cache=True, state=draft_state)[-1]

    round_idx = 0
    while len(new_tokens) < max_new and not stopped:
        round_idx += 1
        room = max_new - len(new_tokens)

        # ---- 1. 草稿逐 token 增量续写，最多 k 个（复用草稿 KV）----
        draft_block: list[int] = []
        next_logits = draft_next_logits
        for _ in range(min(k, room)):
            nxt = greedy_argmax(next_logits)   # 首个 token 不额外触发前向
            draft_block.append(nxt)
            if nxt == tokenizer.EOS_ID or len(draft_block) == min(k, room):
                break
            next_logits = draft.forward(
                [nxt], use_cache=True, state=draft_state)[-1]

        # ---- 2. 目标仅 1 次前向整块核验（复用目标 KV，不重算前缀）----
        if pending is None:
            target_input = list(seq) + draft_block       # 首轮含完整提示词
            pred_start = len(seq) - 1                    # block 预测起始行
        else:
            target_input = [pending] + list(draft_block)  # 后续轮：pending+新块
            pred_start = 0
        logits = target.forward(
            target_input, use_cache=True, state=target_state)
        target_preds = [
            greedy_argmax(logits[pred_start + i])
            for i in range(len(draft_block))
        ]

        # ---- 3. 逐位核验，求接受数 ----
        accepted = 0
        for d, t in zip(draft_block, target_preds):
            if d == t:
                accepted += 1
            else:
                break

        # ---- 4. 组装本轮确认 token ----
        accepted_tokens = draft_block[:accepted]
        if accepted < len(draft_block):
            decision = "reject"
            tail = target_preds[accepted]          # 目标纠正 token
        else:
            decision = "all_accepted"
            tail = greedy_argmax(logits[-1])       # 整块接受后的目标补词
        round_new = accepted_tokens + [tail]

        # 上限截断（与基线一致：最多 max_new 个新 token）
        round_new = round_new[:room]

        # EOS 截断（与基线一致：EOS 计入输出，其后不再生成）
        eos_pos = next((i for i, t in enumerate(round_new)
                        if t == tokenizer.EOS_ID), None)
        if eos_pos is not None:
            round_new = round_new[:eos_pos + 1]
            stopped = True

        # ---- 5. 更新确认序列 ----
        seq.extend(round_new)
        new_tokens.extend(round_new)

        # ---- 6. 目标 KV 缓存回滚 ----
        # 前向后目标缓存长度 = 旧缓存 + len(target_input)：
        #   首轮 = len(prompt) + len(block)；
        #   后续 = (确认前缀长度-1) + 1(pending) + len(block)
        #          = 旧确认长度 + len(block)。
        # 回滚到"接受前缀"（不含 tail）：
        #   首轮 keep = len(prompt) + accepted；
        #   后续 keep = 旧确认长度 + accepted。
        old_confirmed = len(seq) - len(round_new)
        keep = old_confirmed + accepted
        target_state.cache.truncate(keep)
        # tail 成为下一轮的 pending；若它已被上限/EOS 截掉则无下一轮
        pending = tail if not stopped and len(new_tokens) < max_new else None

        # ---- 7. 草稿 KV 对齐到完整确认前缀 ----
        if draft_state.cache.cache_len > len(seq):
            draft_state.cache.truncate(len(seq))
        # 草稿可能落后（block 提前遇 EOS），逐 token 增量补齐确认前缀；
        # 最后一次前向的输出即下一轮首个草稿 token 的预测。
        while draft_state.cache.cache_len < len(seq):
            draft_next_logits = draft.forward(
                [seq[draft_state.cache.cache_len]],
                use_cache=True, state=draft_state)[-1]

        traces.append(RoundTrace(
            index=round_idx,
            draft_tokens=list(draft_block),
            target_predictions=list(target_preds),
            accepted=accepted,
            decision=decision,
            target_tokens=list(round_new),
            target_forward_input_len=len(target_input),
        ))

    return DecodeResult(
        prompt_ids=list(prompt_ids),
        new_tokens=new_tokens,
        target_forward_count=target_state.forward_count,
        draft_forward_count=draft_state.forward_count,
        rounds=traces,
        stopped_by_eos=stopped,
        target_state=target_state,
        draft_state=draft_state,
    )


@dataclass
class Comparison:
    """两种方式的对照结果。"""
    match: bool
    first_mismatch_index: int | None
    baseline: DecodeResult
    speculative: DecodeResult


def compare(draft: CausalTransformer, target: CausalTransformer,
            prompt_ids: list[int], k: int, max_new: int) -> Comparison:
    """同一次对照内，基线与推测各自持有独立状态（缓存与计数互不干扰）。"""
    base = greedy_baseline(target, prompt_ids, max_new)
    spec = speculative_decode(draft, target, prompt_ids, k, max_new)
    mismatch: int | None = None
    for i, (a, b) in enumerate(zip(base.new_tokens, spec.new_tokens)):
        if a != b:
            mismatch = i
            break
    if mismatch is None and len(base.new_tokens) != len(spec.new_tokens):
        mismatch = min(len(base.new_tokens), len(spec.new_tokens))
    return Comparison(
        match=mismatch is None,
        first_mismatch_index=mismatch,
        baseline=base,
        speculative=spec,
    )
