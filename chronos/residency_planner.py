"""Next-use residency planning for lazy expert pages (backend-free; no torch / mlx imports).

A lazy store holds fewer expert pages than one forward touches, so the page
to evict is the one needed *latest*.  Decoding is repetitive enough to
predict that order:

* forwards of one kind run the same dispatch sequence (prelude, every loop
  iteration of every recurrent layer, coda), and consecutive tokens route
  similarly, so the previous forward of the same kind is a next-use
  reference;
* in self-speculative decoding the *draft queue* is a better reference for
  the verify forward: each draft token ran the prelude and the first
  ``draft_depth`` iterations, and verify re-runs exactly those dispatches
  for the same token, plus the deeper iterations.

``NextUsePlanner`` records the page-access trace of every forward (fed by
the MoE runtime hooks), builds the next-use table of the reference before
each forward, and ranks pages for eviction:

   -1  selected by the dispatch running now and not executed yet → never
    0  used again later in this forward → evict the one needed latest
    1  not again in this forward, expected in the next one
    2  not expected → evict first; pages the LookaheadRouter predicts
       (:meth:`hint`) go last, then the lowest decayed demand
       (``score ← decay·score + uses`` per forward, the sliding window)

It also proposes a bounded prefetch: pages the reference needs early that
are not resident, at most as many as free slots plus resident pages of
class 2, so a prefetch never displaces a page needed sooner.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

Key = Tuple[int, int]          # (physical MoE layer, loop iteration)

_FAR = 1 << 30


class NextUsePlanner:
    def __init__(self, num_experts: int, *, decay: float = 0.7):
        if not 0.0 <= float(decay) < 1.0:
            raise ValueError("decay must be in [0, 1)")
        self.num_experts = int(num_experts)
        self.decay = float(decay)
        self.score = [0.0] * self.num_experts
        self._order: Dict[Key, int] = {}          # dispatch → execution ordinal (learned once)
        self._prev: Dict[str, List[Tuple[int, int]]] = {}
        self._drafts: List[List[Tuple[int, int]]] = []
        self._kind = "single"
        self._trace: List[Tuple[int, int]] = []
        self._next_use: Dict[int, List[int]] = {}
        self._next_forward: set[int] = set()
        self._here = -1
        self._pending: frozenset = frozenset()
        self._hint: frozenset = frozenset()
        self.stats = {"forwards": 0, "accesses": 0, "queue_verifies": 0}

    # ── forward boundaries ────────────────────────────────────────

    def begin_forward(self, kind: str) -> None:
        """``kind``: ``"single"`` (one-token decode), ``"multi"`` (prefill / verify) or ``"draft"``."""
        self._kind = kind
        self._trace = []
        self._here = -1
        self._pending = frozenset()
        ref = list(self._prev.get(kind, ()))
        if kind == "multi" and self._drafts:
            ref += [a for d in self._drafts for a in d]
            self.stats["queue_verifies"] += 1
        table: Dict[int, List[int]] = {}
        for ordinal, eid in ref:
            table.setdefault(eid, []).append(ordinal)
        for v in table.values():
            v.sort()
        self._next_use = table
        nxt = "draft" if kind == "multi" and "draft" in self._prev else kind
        self._next_forward = {eid for _o, eid in self._prev.get(nxt, ())}

    def access(self, layer: int, iteration: int, eid: int, pending: Iterable[int] = ()) -> None:
        """Dispatch ``(layer, iteration)`` is about to run expert ``eid``.

        ``pending`` are the dispatch's other selected experts that run after
        it (an MoE runs its selected experts in id order); they are never
        evicted to make room for ``eid``.
        """
        key = (int(layer), int(iteration))
        ordinal = self._order.get(key)
        if ordinal is None:
            ordinal = self._order[key] = len(self._order)
        self._here = ordinal
        self._pending = frozenset(int(e) for e in pending)
        entry = (ordinal, int(eid))
        if not self._trace or self._trace[-1] != entry:      # loader + touch report one run
            self._trace.append(entry)
            self.stats["accesses"] += 1

    def end_forward(self, *, verify: bool = False) -> None:
        trace = self._trace
        self._prev[self._kind] = trace
        if self._kind == "draft":
            self._drafts.append(trace)
        elif verify:
            self._drafts = []
        used = {eid for _o, eid in trace}
        d = self.decay
        self.score = [s * d + (1.0 if i in used else 0.0) for i, s in enumerate(self.score)]
        self.stats["forwards"] += 1

    def hint(self, expert_ids: Iterable[int]) -> None:
        """Pages predicted for the next token (LookaheadRouter); kept over other class-2 pages."""
        self._hint = frozenset(int(e) for e in expert_ids)

    # ── ranking ───────────────────────────────────────────────────

    def evict_key(self, eid: int):
        """Sort key; the *largest* is evicted first."""
        if int(eid) in self._pending:
            return (-1, 0, 0.0)
        later = self._next_use.get(int(eid))
        if later:
            for ordinal in later:
                if ordinal > self._here:
                    return (0, ordinal, 0.0)
        if int(eid) in self._next_forward:
            return (1, 0, -self.score[int(eid)])
        return (2, 0 if int(eid) in self._hint else 1, -self.score[int(eid)])

    def victim(self, candidates: Iterable[int]) -> Optional[int]:
        best, best_key = None, None
        for eid in candidates:
            key = self.evict_key(eid)
            if best_key is None or key > best_key:
                best, best_key = eid, key
        return best

    def upcoming(self, limit: Optional[int] = None) -> List[int]:
        """Pages the upcoming forward is expected to touch, earliest first."""
        first = sorted((v[0], eid) for eid, v in self._next_use.items() if v)
        ids = [eid for _o, eid in first]
        return ids if limit is None else ids[:max(0, int(limit))]

    def prefetch_plan(self, resident: Iterable[int], capacity: int) -> List[int]:
        """Missing pages the upcoming forward needs, earliest first, that fit without
        displacing a page needed sooner: free slots plus resident class-2 pages.

        Call between :meth:`begin_forward` and the forward itself.
        """
        resident = set(int(e) for e in resident)
        spare = sum(1 for e in resident if self.evict_key(e)[0] == 2)
        budget = spare + max(0, int(capacity) - len(resident))
        if budget <= 0:
            return []
        return [eid for eid in self.upcoming() if eid not in resident][:budget]
