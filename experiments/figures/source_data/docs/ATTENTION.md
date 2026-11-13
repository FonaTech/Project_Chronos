# Attention

## 1. Multi-head latent attention (MLA)

A standard attention block stores $H \cdot d_h$ bytes per token in its KV
cache; for $H = 16$ and $d_h = 64$ that is 1 KiB per token per layer. MLA
replaces the per-head projections with a low-rank factorisation
$W = W^\mathrm{up} W^\mathrm{down}$ and stores only the latent state
$W^\mathrm{down} x \in \mathbb{R}^{d_\ell}$; the cache is $d_\ell$ bytes
per token per layer, where $d_\ell \ll H \cdot d_h$ (we use $d_\ell = 96$
at $H = 16, d_h = 64$).

### Formulas

Let $d_\ell$ be the latent dimension and $d_r$ the RoPE head dimension.

$$
\begin{aligned}
c^{KV}_t &= W_{DKV}\, x_t \in \mathbb{R}^{d_\ell}, & \text{cached}\\
c^{Q}_t  &= W_{DQ}\, x_t  \in \mathbb{R}^{d_\ell}, & \text{per step} \\
k^C_t    &= W_{UK} c^{KV}_t \in \mathbb{R}^{H \cdot d_h}, \\
v_t      &= W_{UV} c^{KV}_t \in \mathbb{R}^{H \cdot d_h}, \\
q^C_t    &= W_{UQ} c^{Q}_t  \in \mathbb{R}^{H \cdot d_h}, \\
k^R_t    &= \mathrm{RoPE}(W_{KR} x_t), \quad q^R_t = \mathrm{RoPE}(W_{QR} c^Q_t),
\end{aligned}
$$

and the attention logits are

$$
A_{ts} \propto \langle [q^C_t; q^R_t],\ [k^C_s; k^R_s]\rangle.
$$

Only $c^{KV}_t$ and $k^R_t$ enter the cache. Configuration:
`kv_latent_dim`, `rope_dim`, `rope_layout`.

### RoPE layouts

| `rope_layout` | Behaviour | Where to use it |
|---|---|---|
| `per_head` | RoPE applied in each head's own frame | **Default**; shift-invariant, document packing matches an isolated document to 1.6e-3. |
| `legacy_sliced` | 0.3.0 reference layout | Kept for old 50M checkpoints; _not_ shift-invariant (rel 0.2 across packings), so document masking does not match. |
| `decoupled` | RoPE on a dedicated $d_r$-dim head | MLA default head. |

### Blockwise MLA

The blockwise path (`attention_block_size`, 0.3.2a1) streams the query
block by block and fuses softmax online (like FlashAttention). The KV
tensor is materialised once in bf16; each query block builds a
$B$-row$\times$block-length score panel in registers. Peak memory per
layer is bounded by

$$
\mathrm{Peak}_\mathrm{bwa} \approx B \cdot (T \cdot d_\ell + b \cdot T),
$$

where $b$ is the block size (default 1024). A full-mask baseline at the
same shape is dominated by the $B \cdot H \cdot T^2$ score buffer. Measured
reductions on one layer, bf16, GPU:

| Sequence | Full-causal MLA | Blockwise MLA |
|---|---|---|
| 16384 | 10.3 GiB  | 0.96 GiB |
| 32768 | 39.5 GiB  | 1.41 GiB |
| 65536 | 148.6 GiB | 2.49 GiB |

Exactness: value `rel 0`, grad `rel <= 8.5e-7` vs the full-mask VJP.

## 2. Sliding-window attention (SWA)

Every other block applies attention with a fixed window $W$. SWA keeps
cost linear in sequence length and complements MLA (which handles the
long-range path). The window size is a model constant
(`sliding_window_size`, default 1024); it is not scheduled.

## 3. Attention modes and the recurrent loop

The recurrent core alternates attention types. With the four-layer block
`A B C E` we use:

- A = dense MLA (full causal)
- B = MoE block (no self-attention)
- C = MoE block (no self-attention)
- E = SWA

The KV cache for A and E is built at iteration 0 and reused for every
later iteration (`shared_first`). Alternative `recurrent_attention_mode`
values keep the old 0.3.0 behaviour for comparison:

| Value | KV at iteration $r$ |
|---|---|
| `shared_first` (default) | Built once; later iterations reuse it. |
| `per_iteration` | Rebuilt every iteration (slow, costs memory). |
| `first_only` | Only iteration 0 pays for attention. |

Block order plus `shared_first` gives the fastest loop in our measurements
and keeps self-speculative decoding exact (the draft reads the same KV).

![Block vs layer order](assets/block_vs_layer_schema.png)

## 4. Long-context memory

Beyond blockwise attention, long-context training adds:

- **Chunked CE** (`ce_chunk_size`) — stream the LM head over chunks of
  the sequence to bound the peak activations of the softmax CE. At 64k
  tokens it cuts peak by 8.3 GiB and matches the full path at
  `rel <= 5e-6`.
- **Document masking** (`document_masking`) — a segment-id matrix the
  attention layer reads so packed documents do not attend across
  boundaries. The RoPE path must be `per_head` for packing to match an
  isolated document.
- **MoE chunked tokens** (`moe_chunk_tokens`) — dispatches experts in
  chunks under `mx.checkpoint`, so the sparse scatter never materialises
  a $B T \times N_E$ gate activation.
- **RoPE scaling** (`rope_train_scaling`) — yarn / linear / NTK schedules,
  Torch and MLX tables agree to < 2e-5.

![Long-context memory](assets/long_context_memory.png)

Numbers: `reports/long_context_20261109/real_corpus_ab.json`; chapter
[LONG_CONTEXT.md](LONG_CONTEXT.md) has the full table.

## 5. Code pointers

| File | What it holds |
|---|---|
| `chronos/mlx/model.py` | MLA, SWA, loop structure. |
| `chronos/mlx/blockwise_attention.py` | Blockwise MLA / SWA kernel. |
| `chronos/mlx/chunked_ce.py` | Chunked LM head cross-entropy. |
| `chronos/mlx/attention.py` | `shared_first`, `per_iteration`, `first_only` KV override. |
| `chronos/model/kernel.py` | Torch reference MLA + SWA. |
| `tests/test_long_context.py` | 14 exactness tests for the long-context path. |
| `tests/test_recurrent_order.py` | 18 tests for block vs layer order. |
