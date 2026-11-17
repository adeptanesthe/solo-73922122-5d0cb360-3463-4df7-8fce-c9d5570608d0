"""关键功能测试：
  1. 带缓存与无缓存两条前向路径对同一序列给出逐位相同的 logits；
  2. 推测解码与无缓存贪心基线在多组提示词/k/max_new 下逐 token 完全一致；
  3. 目标前向次数 == 推测轮数，且严格少于基线（存在接受时）；
  4. KV 缓存长度不变量：目标每轮只吃新块，不重算已缓存前缀；
  5. EOS 截断、max_new 上限、平局取小 id 的行为正确；
  6. 预设提示词确实分别覆盖 reject 与 all_accepted；
  7. 并发隔离：每请求独立 KV 缓存/前向计数，停顿后恢复与单独运行完全一致，
     含逐轮接受数与调用次数；HTTP 层多请求并发结果与串行结果逐一相同。
"""

from __future__ import annotations

import os
import sys
import threading

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import tokenizer
from app.decoding import compare, greedy_baseline, speculative_decode
from app.model import ModelState, get_models, greedy_argmax

torch.set_num_threads(1)
DRAFT, TARGET = get_models()

PROMPTS = [
    "the quick brown fox jumps over ",
    "hello world, this is a demo of ",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ",
    "x = 1\ny = 2\nprint(x + y)  # ",
    "0 1 1 2 3 5 8 13 21 34 next ",
    "speculative decoding verifies that ",
    "once upon a time there lived ",
]

# 并发测试用的两条提示词（分别覆盖整块接受与拒绝两种路径）
CONCURRENT_PROMPTS = [
    "the quick brown fox jumps over the lazy dog while ",
    "python java rust go c plus plus ruby swift scala ",
]


def test_cached_vs_uncached_logits_identical() -> None:
    target = TARGET
    ids = tokenizer.encode("the quick brown fox ")
    # 无缓存整序列前向
    full_state = ModelState(target)
    full = target.forward(ids, use_cache=False, state=full_state)
    # 缓存路径：先喂前缀，再喂最后一个 token
    inc_state = ModelState(target)
    target.forward(ids[:-1], use_cache=True, state=inc_state)
    inc = target.forward([ids[-1]], use_cache=True, state=inc_state)
    # float32 矩阵乘法结合顺序不同会有 ~1e-5 级舍入误差
    assert torch.allclose(full[-1], inc[-1], atol=1e-4), "两条前向路径 logits 不一致"
    # 关键：argmax 贪心决策必须完全相同
    assert full[-1].argmax() == inc[-1].argmax()

    # 多 token 增量块也必须与整序列前向逐位一致
    block_state = ModelState(target)
    target.forward(ids[:3], use_cache=True, state=block_state)
    block = target.forward(ids[3:], use_cache=True, state=block_state)
    assert torch.allclose(full[3:], block, atol=1e-4), "多 token 增量前向与整序列不一致"
    assert torch.equal(full[3:].argmax(dim=-1), block.argmax(dim=-1))
    # 两个请求状态各自独立计数，互不影响
    assert inc_state.forward_count == 2 and block_state.forward_count == 2
    assert full_state.forward_count == 1
    print("ok: 缓存/无缓存前向 logits 逐位一致（argmax 决策完全相同）")


def test_cache_rollback_no_contamination() -> None:
    """拒绝场景核心保证：缓存回滚后换续写，预测与对应整序列前向完全一致。"""
    target = TARGET
    ids = tokenizer.encode("the quick brown fox jumps over ")
    block = [10, 20, 30, 40]

    full_state = ModelState(target)
    full = target.forward(ids + block, use_cache=False, state=full_state)
    ref_preds = [greedy_argmax(full[len(ids) + j]) for j in range(len(block))]

    state = ModelState(target)
    target.forward(ids, use_cache=True, state=state)
    l1 = target.forward(block[:2], use_cache=True, state=state)
    l2 = target.forward(block[2:], use_cache=True, state=state)
    cached_preds = [greedy_argmax(x) for x in list(l1) + list(l2)]
    assert cached_preds == ref_preds

    # 回滚到提示词长度后换一组续写（模拟拒绝后用目标 token 纠正）
    state.cache.truncate(len(ids))
    new_block = [50, 60]
    l3 = target.forward(new_block, use_cache=True, state=state)
    ref2 = target.forward(ids + new_block, use_cache=False,
                          state=ModelState(target))
    assert greedy_argmax(l3[0]) == greedy_argmax(ref2[len(ids)])
    assert greedy_argmax(l3[1]) == greedy_argmax(ref2[len(ids) + 1])
    print("ok: 缓存回滚后续写无污染，与整序列前向逐位一致")


def test_spec_matches_greedy_all_cases() -> None:
    n_cases = 0
    for p in PROMPTS:
        ids = tokenizer.encode(p)
        for k in (1, 2, 3, 4):
            for max_new in (1, 7, 32):
                r = compare(DRAFT, TARGET, ids, k, max_new)
                assert r.match, (
                    f"不一致: prompt={p!r} k={k} max_new={max_new} "
                    f"idx={r.first_mismatch_index} "
                    f"base={r.baseline.new_tokens} spec={r.speculative.new_tokens}"
                )
                assert len(r.speculative.new_tokens) <= max_new
                n_cases += 1
    print(f"ok: {n_cases} 组参数下推测解码与贪心基线逐 token 一致")


def test_target_forward_count_equals_rounds_and_saves() -> None:
    ids = tokenizer.encode(PROMPTS[0])
    r = compare(DRAFT, TARGET, ids, 4, 32)
    spec = r.speculative
    assert spec.target_forward_count == len(spec.rounds), "目标前向次数必须等于轮数"
    # 每轮恰好一次目标前向，输入长度 = 首轮提示词+块，后续仅块
    first = spec.rounds[0]
    assert first.target_forward_input_len == len(ids) + len(first.draft_tokens)
    for rd in spec.rounds[1:]:
        assert rd.target_forward_input_len == 1 + len(rd.draft_tokens)
    # 32 个新 token、k=4，推测轮数必然显著少于 32
    assert spec.target_forward_count < r.baseline.target_forward_count
    print(f"ok: 目标前向 {spec.target_forward_count} 次 vs 基线 "
          f"{r.baseline.target_forward_count} 次，轮数=前向数")


def test_cache_prefix_not_recomputed() -> None:
    """缓存长度不变量（pending-tail 方案）：
      - 草稿缓存始终对齐完整确认序列；
      - 目标缓存 == 确认序列长度 - 1（正常轮，tail 待下一轮送入），
        终止轮可能等于确认长度（tail 被上限/EOS 截断，无 pending）；
      - 后续轮目标前向输入恒为 1 个 pending + 草稿块，绝不包含已缓存前缀。
    """
    ids = tokenizer.encode(PROMPTS[1])
    res = speculative_decode(DRAFT, TARGET, ids, 3, 20)
    confirmed = len(ids) + len(res.new_tokens)
    assert res.draft_state.cache.cache_len == confirmed
    assert res.target_state.cache.cache_len in (confirmed - 1, confirmed)
    for rd in res.rounds[1:]:
        assert rd.target_forward_input_len == 1 + len(rd.draft_tokens)
    assert res.rounds[0].target_forward_input_len == len(ids) + len(
        res.rounds[0].draft_tokens)

    ids2 = tokenizer.encode(PROMPTS[2])
    res2 = speculative_decode(DRAFT, TARGET, ids2, 2, 10)
    confirmed2 = len(ids2) + len(res2.new_tokens)
    assert res2.draft_state.cache.cache_len == confirmed2
    assert res2.target_state.cache.cache_len in (confirmed2 - 1, confirmed2)
    print("ok: KV 缓存不变量成立，已缓存前缀未被重算")


def test_max_new_hard_cap() -> None:
    ids = tokenizer.encode(PROMPTS[0])
    for cap in (1, 2, 5, 32):
        r = compare(DRAFT, TARGET, ids, 4, cap)
        assert len(r.baseline.new_tokens) == len(r.speculative.new_tokens) <= cap
        assert r.match
    print("ok: max_new 上限在两种方式下行为一致")


def test_eos_handling() -> None:
    """EOS 处理：自然场景下位置一致；再用 hook 强制注入 EOS 覆盖所有 k/上限组合。"""
    ids = tokenizer.encode("the quick brown fox ")
    base = greedy_baseline(TARGET, ids, 32)
    spec = speculative_decode(DRAFT, TARGET, ids, 4, 32)
    base_eos = [i for i, t in enumerate(base.new_tokens) if t == tokenizer.EOS_ID]
    spec_eos = [i for i, t in enumerate(spec.new_tokens) if t == tokenizer.EOS_ID]
    assert base_eos == spec_eos

    # 强制目标把 EOS 作为最大 logit，验证两条路径 EOS 截断位置逐位一致
    def eos_hook(module, inp, out):
        out2 = out.clone()
        out2[..., tokenizer.EOS_ID] = out2.max().item() + 100.0
        return out2

    handle = TARGET.lm_head.register_forward_hook(eos_hook)
    try:
        for k in (1, 2, 3, 4):
            for cap in (8, 16, 32):
                r = compare(DRAFT, TARGET, ids, k, cap)
                assert r.match, (k, cap, r.first_mismatch_index)
                be = [i for i, t in enumerate(r.baseline.new_tokens)
                      if t == tokenizer.EOS_ID]
                se = [i for i, t in enumerate(r.speculative.new_tokens)
                      if t == tokenizer.EOS_ID]
                assert be == se and len(be) == 1
                assert r.baseline.stopped_by_eos and r.speculative.stopped_by_eos
                assert len(r.baseline.new_tokens) == be[0] + 1
    finally:
        handle.remove()
    print("ok: EOS 截断在自然与强制场景下两种方式完全一致")


def test_tie_prefers_small_id() -> None:
    logits = torch.zeros(5)  # 全部并列
    from app.model import greedy_argmax
    assert greedy_argmax(logits) == 0
    logits[3] = 1.0
    assert greedy_argmax(logits) == 3
    print("ok: 平局时取词表序号小者")


def test_round_traces_contain_both_decisions() -> None:
    """固定预设必须分别触发 reject 与 all_accepted（网页两个预设按钮的依据）。"""
    from app.main import PRESET_ALL_ACCEPTED, PRESET_REJECT

    r_reject = compare(DRAFT, TARGET, tokenizer.encode(PRESET_REJECT), 4, 32)
    assert "reject" in {rd.decision for rd in r_reject.speculative.rounds}, \
        "拒绝预设未触发 reject"
    assert r_reject.match

    r_accept = compare(DRAFT, TARGET, tokenizer.encode(PRESET_ALL_ACCEPTED), 4, 32)
    assert "all_accepted" in {rd.decision for rd in r_accept.speculative.rounds}, \
        "整块接受预设未触发 all_accepted"
    assert r_accept.match
    print(f"ok: 拒绝预设出现拒绝轮（前向 {r_reject.speculative.target_forward_count} 次）")
    print(f"    整块接受预设（前向 {r_accept.speculative.target_forward_count} 次）")


def test_deterministic_weights() -> None:
    """重新构建模型应得到完全相同的权重（固定种子）。"""
    from app.model import _build_pair
    pair1 = _build_pair()
    pair2 = _build_pair()
    for m1, m2 in zip(pair1, pair2):
        for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
            assert n1 == n2 and torch.equal(p1, p2)
    print("ok: 固定种子下权重确定可复现")


# ---------------------------------------------------------------------------
# 并发隔离测试
# ---------------------------------------------------------------------------

def _solo_reference(ids: list[int], k: int = 4, max_new: int = 32):
    """单独（无并发）运行一次完整对照，作为并发请求的期望值。"""
    return compare(DRAFT, TARGET, ids, k, max_new)


def _assert_same_as_solo(got, ref, label: str) -> None:
    assert got.match, f"{label}: 对照判定不是 PASS"
    assert got.speculative.new_tokens == ref.speculative.new_tokens, \
        f"{label}: 推测输出与单跑不一致"
    assert got.baseline.new_tokens == ref.baseline.new_tokens, \
        f"{label}: 基线输出与单跑不一致"
    assert got.speculative.target_forward_count == \
        ref.speculative.target_forward_count, \
        f"{label}: 目标前向次数 {got.speculative.target_forward_count} != " \
        f"{ref.speculative.target_forward_count}"
    assert got.speculative.draft_forward_count == \
        ref.speculative.draft_forward_count, \
        f"{label}: 草稿前向次数 {got.speculative.draft_forward_count} != " \
        f"{ref.speculative.draft_forward_count}"
    assert got.baseline.target_forward_count == \
        ref.baseline.target_forward_count, f"{label}: 基线前向次数漂移"
    # 逐轮接受数与判定也必须与单跑完全一致
    got_rounds = [(rd.accepted, rd.decision, rd.target_tokens,
                   rd.target_forward_input_len) for rd in got.speculative.rounds]
    ref_rounds = [(rd.accepted, rd.decision, rd.target_tokens,
                   rd.target_forward_input_len) for rd in ref.speculative.rounds]
    assert got_rounds == ref_rounds, f"{label}: 逐轮接受数/判定与单跑不一致"
    assert got.speculative.stopped_by_eos == ref.speculative.stopped_by_eos


def test_concurrent_pause_after_draft_preheat() -> None:
    """题目核心场景：A 在草稿预热后停顿，B 先完整跑完，A 恢复后结果、
    逐轮接受数、前向次数均与单独运行一致。"""
    from app.model import CausalTransformer

    ids_a = tokenizer.encode(CONCURRENT_PROMPTS[0])
    ids_b = tokenizer.encode(CONCURRENT_PROMPTS[1])
    ref_a = _solo_reference(ids_a)
    ref_b = _solo_reference(ids_b)

    warmed = threading.Event()
    b_done = threading.Event()
    started = threading.Event()
    a_draft_calls = [0]
    orig_forward = CausalTransformer.forward

    def pausing_forward(self, token_ids, use_cache, state=None):
        # A 的第 2 次草稿前向 = 草稿预热（第 1 次）完成之后；在此停顿，
        # 让 B 完整跑完。旧实现会让 A 在此读到 B 写入的共享 KV 缓存。
        if (self is DRAFT and started.is_set() and not warmed.is_set()
                and state is not None):
            a_draft_calls[0] += 1
            if a_draft_calls[0] == 2:
                warmed.set()
                b_done.wait()
        return orig_forward(self, token_ids, use_cache, state)

    CausalTransformer.forward = pausing_forward
    results: dict[str, object] = {}
    try:
        def run_a() -> None:
            started.set()
            results["a"] = compare(DRAFT, TARGET, ids_a, 4, 32)
            warmed.set()  # 兜底，避免停顿点未命中时死锁

        def run_b() -> None:
            warmed.wait()
            results["b"] = compare(DRAFT, TARGET, ids_b, 4, 32)
            b_done.set()

        t_a = threading.Thread(target=run_a, name="request-A")
        t_b = threading.Thread(target=run_b, name="request-B")
        t_a.start()
        t_b.start()
        t_a.join(timeout=60)
        t_b.join(timeout=60)
        assert not t_a.is_alive() and not t_b.is_alive(), "并发线程意外阻塞"
    finally:
        CausalTransformer.forward = orig_forward

    assert a_draft_calls[0] >= 2, "测试停顿点未命中，场景未被实际执行"
    _assert_same_as_solo(results["a"], ref_a, "停顿请求 A")
    _assert_same_as_solo(results["b"], ref_b, "先完成请求 B")
    print("ok: A 预热后停顿、B 先完成，恢复后 A/B 的输出、逐轮接受数、"
          "前向次数均与单独运行一致")


def test_concurrent_many_requests_isolated() -> None:
    """无锁压力场景：多线程同时跑不同提示词/参数，每个结果都必须与
    自己单独运行的结果逐 token、逐轮、逐计数一致（拒绝回滚与整块接受
    路径都覆盖）。"""
    jobs = []
    for p in CONCURRENT_PROMPTS + PROMPTS[:4]:
        ids = tokenizer.encode(p)
        for k in (1, 2, 3, 4):
            for max_new in (7, 32):
                jobs.append((p, ids, k, max_new))

    refs = {(p, k, m): _solo_reference(ids, k, m) for p, ids, k, m in jobs}
    results: dict[tuple, object] = {}
    errors: list[BaseException] = []

    def worker(job) -> None:
        p, ids, k, m = job
        try:
            results[(p, k, m)] = compare(DRAFT, TARGET, ids, k, m)
        except BaseException as exc:  # noqa: BLE001 - 测试线程需收集全部异常
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(job,)) for job in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
        assert not t.is_alive(), "并发任务超时"
    assert not errors, f"并发请求出现异常: {errors!r}"
    assert len(results) == len(jobs)
    for job in jobs:
        _assert_same_as_solo(results[(job[0], job[2], job[3])],
                             refs[(job[0], job[2], job[3])],
                             f"并发 {job[0]!r} k={job[2]} m={job[3]}")
    print(f"ok: {len(jobs)} 个并发请求（含拒绝与整块接受）全部与单跑结果一致")


def test_http_concurrent_requests_match_serial() -> None:
    """HTTP 层并发：FastAPI 线程池中同时处理多个 /api/decode，
    返回结果与串行逐一相同（真实服务路径，含每请求状态隔离）。"""
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    payloads = [
        {"prompt": p, "k": k, "max_new": m}
        for p in CONCURRENT_PROMPTS
        for k in (1, 3, 4)
        for m in (8, 32)
    ]
    serial = []
    for body in payloads:
        r = client.post("/api/decode", json=body)
        assert r.status_code == 200, r.text
        serial.append(r.json())

    http_errors: list[BaseException] = []

    def post(body, box, idx) -> None:
        try:
            with TestClient(app) as c:
                resp = c.post("/api/decode", json=body)
                assert resp.status_code == 200, resp.text
                box[idx] = resp.json()
        except BaseException as exc:  # noqa: BLE001
            http_errors.append(exc)

    parallel: list[dict | None] = [None] * len(payloads)
    threads = [
        threading.Thread(target=post, args=(body, parallel, i))
        for i, body in enumerate(payloads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
        assert not t.is_alive(), "HTTP 并发请求超时"
    assert not http_errors, f"HTTP 并发异常: {http_errors!r}"

    for body, got, want in zip(payloads, parallel, serial):
        assert got is not None
        assert got["match"] and want["match"]
        assert got["baseline"]["new_tokens"] == want["baseline"]["new_tokens"]
        assert got["speculative"]["new_tokens"] == want["speculative"]["new_tokens"]
        assert (got["speculative"]["target_forward_count"]
                == want["speculative"]["target_forward_count"])
        assert (got["speculative"]["draft_forward_count"]
                == want["speculative"]["draft_forward_count"])
        got_rounds = [(rd["accepted"], rd["decision"])
                      for rd in got["speculative"]["rounds"]]
        want_rounds = [(rd["accepted"], rd["decision"])
                       for rd in want["speculative"]["rounds"]]
        assert got_rounds == want_rounds, body
    print(f"ok: HTTP 层 {len(payloads)} 个并发请求与串行结果逐 token/逐轮/计数一致")


if __name__ == "__main__":
    test_deterministic_weights()
    test_cached_vs_uncached_logits_identical()
    test_cache_rollback_no_contamination()
    test_tie_prefers_small_id()
    test_spec_matches_greedy_all_cases()
    test_target_forward_count_equals_rounds_and_saves()
    test_cache_prefix_not_recomputed()
    test_max_new_hard_cap()
    test_eos_handling()
    test_round_traces_contain_both_decisions()
    test_concurrent_pause_after_draft_preheat()
    test_concurrent_many_requests_isolated()
    test_http_concurrent_requests_match_serial()
    print("\n全部测试通过 ✔")
