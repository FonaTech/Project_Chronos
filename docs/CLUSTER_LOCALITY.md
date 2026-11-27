# Cluster locality: a fixed partition and learned routing

Cluster locality connects training-time routing to the expert-page working set. The implemented objective changes how token streams distribute routing probability over a **fixed expert-ID partition**. It does not move expert objects in memory, learn cluster boundaries, or directly minimize a measured SSD latency.

This distinction gives the mechanism three separate objects:

1. **Partition:** a deterministic map from expert IDs to storage groups.
2. **Routing specialization:** a learned dependence between stream content and those groups.
3. **Runtime paging:** a backend-specific policy that loads and retains expert weights.

[MOE_ROUTING.md](MOE_ROUTING.md) defines the per-token expert function. [RESIDENCY.md](RESIDENCY.md) and [LAZY_EXPERTS.md](LAZY_EXPERTS.md) specify paging. The present chapter derives the statistical objective between them.

![Cluster-locality coefficient sweep](assets/cluster_locality_sweep.png)

*Recorded coefficient sweep on a synthetic stream task. CE, cross-entropy. Cluster reads are software runtime counters per generated token, not physical storage bandwidth or a universal speedup. The final-curve CE difference is distinct from the separate held-out evaluation.*

## 1. Fixed membership and the dispatch axis

Let $E$ be the routed-expert count and $N_c$ the number of clusters. Membership is

$$
 \pi:\{0,\ldots,E-1\}\longrightarrow\{0,\ldots,N_c-1\},\qquad
 M_{e,c}=\mathbf1\{\pi(e)=c\}.
$$

The current training path obtains $M$ from `training_clusters(config)`. It splits expert IDs into contiguous, near-equal groups. The cluster count is the explicit positive `cluster_locality_clusters`, or otherwise

$$
 N_c=\mathrm{clip}_{[1,E]}\left(\left\lceil\frac{E}{2k}\right\rceil\right).
$$

There is no implemented `clusters.partition_json` training field in this snapshot. A cluster manifest used by storage is a separate object. [Partition implementation](../chronos/expert_clusters.py#L28).

Every physical MoE layer has its own expert weights. Applying the same expert-ID partition across layers does not tie those weights. It aligns group labels used by the objective and by packed storage.

The trainer supplies router probabilities

$$
 Q\in\mathbb R^{B\times T\times V\times E},\qquad
 \sum_eQ_{b,t,v,e}=1,
$$

where $V$ indexes the collected dispatches, including repeated calls of recurrent layers. Since 0.3.5 the default $Q$ is the **dispatch mass**: each dispatch keeps its $k$ selected experts, renormalized, and the probabilities come from the gate applied to a gradient-free copy of the MoE input (section 9). With `cluster_locality_mass="soft"` and `cluster_input_detach=False`, $Q$ is the full pre-top-k softmax of the live router, which every release before 0.3.5 used. [Collection and loss assembly](../chronos/trainer/loss_mixin.py#L218), [source selection](../chronos/trainer/loss_mixin.py#L102).

For an equal-length window $w$ with $n_w$ tokens,

$$
 p(c\mid w)=\frac{1}{n_wV}
   \sum_{t\in w}\sum_{v=1}^{V}\sum_{e=1}^{E}Q_{t,v,e}M_{e,c}.
$$

The dispatch average matters: locality is encouraged across tokens and collected layer/iteration calls, not only inside one token's selected set. [Window pooling](../chronos/model/temporal_loss.py#L70).

```mermaid
flowchart TB
    IDs["Expert IDs and configured<br/>cluster count"] --> P["Contiguous fixed partition"]
    P --> M["One-hot membership: E × N_c"]
    Q["Soft router probabilities: B ×<br/>T × V × E"] --> Pool["Pool tokens and dispatches"]
    M --> Pool
    Pool --> PW["One cluster distribution per<br/>stream window"]
    PW --> MI["Budget objective (default)<br/>or mutual information"]
    MI -. "Gradient updates the gate weights;<br/>attached mode also upstream representations" .-> Q
```
*Training objects. Cluster boundaries remain fixed. The learned probabilities change; the arrows do not depict expert migration.*

## 2. Deriving the mutual-information objective

Let $\alpha_w$ be the empirical weight of window $w$, with $\sum_w\alpha_w=1$. For equal unmasked windows, the implementation uses equal weights. Define the marginal

$$
 \bar p_c=\sum_w\alpha_wp(c\mid w).
$$

Entropy and conditional entropy, using natural logarithms, are

$$
 H(C)=-\sum_c\bar p_c\log\bar p_c,
 \qquad
 H(C\mid W)=-\sum_w\alpha_w\sum_cp(c\mid w)\log p(c\mid w).
$$

Their difference is mutual information:

$$
 I(W;C)=H(C)-H(C\mid W)
       =\sum_w\alpha_w\sum_cp(c\mid w)
         \log\frac{p(c\mid w)}{\bar p_c}.
$$

The second expression is the weighted mean of $D_{\mathrm{KL}}(p(C\mid w)\|\bar p)$. It is nonnegative for valid probability distributions. Since $I(W;C)\le H(C)\le\log N_c$,

$$
 L_{\mathrm{cl}}=1-\frac{I(W;C)}{\log N_c}
 =1-\frac{H(C)}{\log N_c}+\frac{H(C\mid W)}{\log N_c}
 \in[0,1],\qquad N_c>1.
$$

This last decomposition explains the two pressures:

- Reducing $H(C\mid W)$ concentrates each stream in few clusters.
- Increasing $H(C)$ preserves diversity across the batch's streams.

A single global popular cluster does not solve the objective: both entropies then vanish and $L_{\mathrm{cl}}=1$. Conversely, $L_{\mathrm{cl}}=0$ requires deterministic per-window assignments **and a uniform cluster marginal**. Deterministic assignments alone do not suffice. Finite-batch composition may make this ideal unattainable.

[Exact Torch loss](../chronos/model/temporal_loss.py#L106), [MLX counterpart](../chronos/mlx/training/trainer.py#L277).

Since 0.3.5 this objective is selected with `cluster_locality_objective="mutual_info"`. The default is the budget objective of section 9, because the mutual-information optimum confines each stream to one cluster and, in the 0.3.5 experiments, that confinement cost language-model loss.

```mermaid
flowchart TD
    PW["Cluster distribution per window"] --> Conditional["Mean within-window entropy"]
    PW --> Marginal["Across-window marginal"]
    Marginal --> H["Marginal entropy"]
    Conditional --> Low["Prefer concentrated streams"]
    H --> High["Prefer diverse streams"]
    Low --> Joint["Maximize mutual information"]
    High --> Joint
    Collapse["All streams choose one cluster"] --> Zero["Both entropies vanish: no mutual information"]
    Zero -. "Not the optimum" .-> Joint
```

*Why local concentration needs global diversity. The objective encourages statistical specialization, not a preassigned semantic name for each cluster.*

### Degenerate cases and numerical safeguards

| Case | Current behavior | Interpretation |
|---|---|---|
| $N_c\le1$ | Loss helper returns zero | No nontrivial cluster choice exists; avoid division by $\log1$ |
| One effective unmasked window in the fixed-window loss | $I=0$, hence $L=1$ with no useful contrastive signal | The marginal equals that window distribution |
| One segment in the segment helper | Returns zero by an explicit guard | Same absence of inter-stream information, with a different constant convention |
| Incomplete final fixed window | Torch and MLX pooling truncate to complete windows | The unpooled suffix does not shape this term |
| `cluster_locality_window=0` | Use a whole sequence row as a window | Multiple batch rows can still provide multiple streams |
| Very small probabilities | Entropy calculations clamp before logarithms | The analytic bounds describe ideal probabilities; finite precision can slightly perturb them |

The distinction between a zero loss and a constant loss of one matters when logging objectives, even when both cases provide no useful locality gradient.

## 3. Segment mode and valid-token weighting

The real field is `cluster_locality_mode="segment"`, not `segment_mode=True`. Segment IDs come from token-stream separators through `segment_ids(input_ids, chunk=window)`. Longer documents are divided into bounded parts. The code does not obtain those IDs from the attention-mask tensor. `document_masking` controls attention and is a different mechanism. [Segment construction](../chronos/expert_clusters.py#L196).

For a segment $w$ containing $n_w$ valid tokens, masked pooling is

$$
 p(c\mid w)=\frac{\sum_{t\in w}m_t\sum_v\sum_eQ_{t,v,e}M_{e,c}}
                  {V\sum_{t\in w}m_t},\qquad
 \alpha_w=\frac{\sum_{t\in w}m_t}{\sum_{w'}\sum_{t\in w'}m_t}.
$$

The objective therefore weights segments by valid-token count. Changing from fixed windows to documents changes both the statistical unit and potentially its empirical weighting; it is not merely a renamed boundary list. Padding must not become an artificial “topic.” [Segment objective](../chronos/model/temporal_loss.py#L141).

There is a backend boundary worth retaining in the documentation. MLX's fixed-window helper accepts a valid-token mask and weights windows accordingly. The Torch fixed-window helper in this snapshot does not accept that mask, whereas the Torch segment helper does. Do not claim fully identical masking behavior for every loss mode. [MLX mask path](../chronos/mlx/training/trainer.py#L306), [Torch calls](../chronos/trainer/loss_mixin.py#L232).

```mermaid
flowchart TB
    Tokens["Packed token stream"] --> Sep["Read separator token IDs"]
    Sep --> Docs["Document segments"]
    Docs --> Chunk["Split long documents by<br/>configured limit"]
    Chunk --> Mask["Remove invalid / padded positions"]
    Mask --> Mass["Pool soft routing mass"]
    Mass --> Weight["Weight segments by valid-token<br/>count"]
    Weight --> Loss["Segment mutual-information<br/>objective"]
    Attention["Attention document mask"] -. "Separate mechanism" .-> Note["Does not itself define these<br/>segment IDs"]
```
*Document-aligned locality. Segment formation and attention masking are related configuration choices, but have different data paths.*

## 4. Locality across recurrent iterations

The ordinary cluster objective already pools the dispatch axis. An additional optional term retains the iteration axis explicitly. For recurrent physical group $g$ and iteration $r$, let

$$
 q_{b,t,g,r,c}=\sum_eQ_{b,t,g,r,e}M_{e,c},\qquad
 \bar q_{b,t,g,c}=\frac1R\sum_rq_{b,t,g,r,c}.
$$

The implementation uses mean total variation from the iteration mean:

$$
 L_{\mathrm{iter}}
 =\mathrm{mean}_{b,t,g,r}
    \left[\frac12\sum_c|q_{b,t,g,r,c}-\bar q_{b,t,g,c}|\right],
$$

with a valid-token-weighted mean when a mask is supplied. This is **not** a fixed teacher at iteration zero. It permits different expert IDs inside the same cluster. [Iteration term](../chronos/model/temporal_loss.py#L474).

This term contributes only when the required iteration tensor is available. The Torch collector requires at least two calls and the same recorded call count for every recurrent layer. Prefix-block skipping can invalidate that shape condition; an enabled coefficient does not by itself guarantee a nonzero computed term. [Collector contract](../chronos/trainer/loss_mixin.py#L132).

## 5. What the recorded synthetic sweep establishes

The recorded sweep uses **four topics, 16 routed experts, four clusters, four experts per cluster, 32-token segments, 300 training steps and seed 7**. The model uses MLX with a tiny controlled configuration; it is not the 50M natural-language checkpoint. The benchmark recipe specifies top-2 routing and one shared expert. [Recipe](../tools/bench_cluster_locality.py#L64), [JSON settings](../reports/residency_20261107/sweep/lam_0.3_w32.json).

The table preserves three distinct measurements: final logged curve CE, held-out CE and normalized entropy. All are **measured/reported runtime records on this synthetic task**, with one seed. The entropy field is $H(C\mid W)/\log N_c$, not entropy in nats. [Metric implementation](../tools/bench_cluster_locality.py#L127).

| $\lambda_{\mathrm{cl}}$ | Final curve CE | Held-out CE | $H(C\mid W)/\log N_c$ | Cluster reads/token: window | Cluster reads/token: planned |
|---:|---:|---:|---:|---:|---:|
| 0, baseline | 1.6615 | 1.7034 | 0.4883 | 1.4818 | 1.1146 |
| 0.03 | 1.6517 | 1.7089 | 0.2782 | 0.5130 | 0.3177 |
| 0.1 | 1.6650 | 1.7129 | 0.1508 | 0.1510 | 0.1302 |
| 0.3 | 1.6619 | 1.7116 | 0.0054 | 0.0234 | 0.0234 |
| 1.0 | 1.6689 | 1.7087 | 0.0009 | 0.0260 | 0.0260 |

Sources: [0.03](../reports/residency_20261107/sweep/lam_0.03_w32.json), [0.1](../reports/residency_20261107/sweep/lam_0.1_w32.json), [0.3](../reports/residency_20261107/sweep/lam_0.3_w32.json), [1.0](../reports/residency_20261107/sweep/lam_1.0_w32.json). Each JSON contains a paired baseline; the baseline values above match across these files.

The window-policy read counter changes from 1.4818 to 0.0234 at coefficient 0.3, a derived ratio of about 63.3. This is a reduction in **cluster-load events per generated token under this benchmark**, not a claim of 63.3-fold lower wall-clock latency. The read counter is `cluster_loads/tokens`; a load event can represent different byte volumes across layouts or backends. [Decode accounting](../tools/bench_cluster_locality.py#L158).

The final curve CE and the held-out CE must not be interchanged. For example, the coefficient-0.3 final curve differs from its baseline by 0.0004, while the separate held-out CE differs by 0.0082. Both are differences calculated from the linked records. Neither establishes quality preservation on larger natural-language models or across seeds.

A larger weighted `cluster_term` is not by itself evidence that the loss “crowds out” language modeling. That conclusion needs optimization diagnostics or an ablation; the table alone supports a trade-off observed in this one-seed sweep.

## 6. Fixed training groups versus an offline coactivation layout

`training_clusters` and offline Louvain/greedy grouping are different routes. The former supplies the fixed membership matrix for this objective. The latter derives a storage partition from calibration routing events. Repartitioning storage does not inherently require retraining the model, nor does it mathematically force the original baseline read count. It changes the relation between learned routing and physical layout, which must be measured.

```mermaid
flowchart TD
    Config["Expert count and cluster count"] --> Fixed["Contiguous fixed training partition"]
    Fixed --> MI["Train routing-locality objective"]
    Fixed --> PackA["Pack weights by that partition"]
    Model["A trained model and calibration data"] --> Trace["Collect routing-event log"]
    Trace --> Graph["Build coactivation graph"]
    Graph --> Community["Louvain, or greedy fallback"]
    Community --> PackB["Pack weights by offline partition"]
    PackA --> Runtime["Attach manifest and measure actual paging"]
    PackB --> Runtime
    MI -. "Learned routing distribution" .-> Model
```

*Two layout routes. The diagram does not claim that offline community detection is a compulsory forward-pass stage or that its partition equals the training partition.*

The present collector unions selected expert IDs across physical MoE layers at each token, using each layer's `last_router_probs`, then flattens the sorted per-token unions. The coactivation helper's local neighborhood is a window in this flattened **event list**. It is not automatically a fixed number of original tokens once a token contributes multiple IDs. A recurrent layer's last stored distribution is also not a complete trace of all earlier iterations. [Collector](../chronos/io/cluster_layout.py#L63).

The raw count matrix is symmetrically incremented, then row-normalized. The graph builder consumes upper-triangle entries of that row-normalized matrix as undirected edge weights. Row normalization can break matrix symmetry; this implementation should not be described as a symmetrically normalized similarity metric. [Matrix construction](../chronos/io/cluster_layout.py#L132), [partitioning](../chronos/io/cluster_layout.py#L219).

Storage benefits remain backend-specific. Torch selects requested and budgeted neighboring pages before materialization. The MLX cluster loader loads/evaluates the cluster file before bounded warm-cache admission. Logical page reuse therefore does not establish an identical transient memory or I/O bound on both paths. [Torch page selection](../chronos/io/expert_store.py#L560), [MLX cluster loading](../chronos/mlx/expert_store.py#L290).

## 7. Interpreting an “expert community”

Under a fixed partition, an expert community means a routing association with an existing set of IDs. Topic purity in the synthetic benchmark measures association with known synthetic topics; it is not a measured geometric separation of expert weight vectors or a proof that natural-language domains emerge identically.

Expert parameters continue to learn through the language-model task and other available gradients. The fixed objects are their IDs and this membership map, not frozen expert weights. With `cluster_input_detach=False`, locality can also influence upstream representations that produce router inputs. The 0.3.5 default sends this term's gradient to the gate weights only (section 9). No physical motion or autonomous creation of experts is implemented.

Further, balanced expert use and balanced cluster use are not identical objectives. Near-equal partitions can contain different numbers of experts; a uniform expert marginal then induces cluster mass proportional to group size, whereas the mutual-information normalization favors a uniform cluster marginal at its ideal maximum. This interaction is worth checking when choosing a non-divisible partition.

The next decisive evidence would connect these stages on the **same checkpoint and workload**: routing distributions, selected-expert working set, logical cluster-load count, bytes materialized, peak resident/transient memory, latency and task quality. The existing synthetic sweep supplies part of that chain, not the entire causal argument.

## 8. Actual configuration names

| Field | Class default | Meaning |
|---|---:|---|
| `lambda_cluster_locality` | `0.0` | Compatibility default; enables this training term when positive |
| `cluster_locality_clusters` | `0` | Zero chooses $\lceil E/(2k)\rceil$ |
| `cluster_locality_window` | `32` | Window length; zero means a whole row |
| `cluster_locality_mode` | `"window"` | `"window"` or `"segment"` |
| `cluster_locality_objective` | `"budget"` | `"budget"` (section 9) or `"mutual_info"` (section 2); checkpoints saved before 0.3.5 load as `"mutual_info"` |
| `cluster_locality_budget` | `0` | Clusters per stream $B$ under the budget objective; zero chooses $\lceil N_c/2\rceil$ |
| `cluster_locality_mass` | `"dispatch"` | `"dispatch"`: top-k mass, renormalized; `"soft"`: full softmax |
| `cluster_input_detach` | `True` | The cluster term reads the gate on a gradient-free input and trains the gate weights only |
| `lambda_iteration_cluster` | `0.0` | Optional iteration spread |

[Configuration](../chronos/model/config.py#L279). The planner can select `DEFAULT_LAMBDA=0.3` for eligible MoE training plans, while old checkpoints retain their stored recipe. Thus “class default zero” is not equivalent to “all new training recipes disable locality.” [Planner choice](../chronos/model/planner.py#L566), [shared training helper](../chronos/expert_clusters.py#L153).

`inference_clusters(..., paging="auto")` uses training groups when the stored coefficient is positive and otherwise uses singleton expert pages. This is a **metadata-based decision**, not an independent inspection proving how the weights were trained. Editing a checkpoint coefficient after training does not establish learned locality. [Automatic paging](../chronos/expert_clusters.py#L90).

## 9. The budget objective (0.3.5)

### 9.1 Why the mutual-information objective cost loss

At the mutual-information optimum each stream places all of its routing mass in one cluster. A token then reuses the same few experts at every layer and loop iteration. The 0.3.4 default also trained the trunk through this term: its gradient reached attention, embeddings and experts, not only the gate. 0.3.5 changes three things, each behind a field: the objective (`cluster_locality_objective`), the probabilities it reads (`cluster_locality_mass`) and the path of its gradient (`cluster_input_detach`).

The table isolates these choices on one recipe. All runs use the 15M shape (hidden size 160, 24 routed experts, top-2, one shared expert, loop depth 4; 14.98M total and 6.87M active parameters), MLX on one Apple M5, constant learning rate 1.5e-3, 65,536 tokens per step, seed 1234, segment mode with window 256 and $\lambda_{\mathrm{cl}}=0.3$. The CE gap is the validation CE minus that of the $\lambda_{\mathrm{cl}}=0$ run (5.248) at step 600. Distinct experts per token count the union of selected experts over all 18 dispatches (layers and loop iterations); hitC is the share of the experts used one and two tokens later that lie in the cluster the lookahead predictor ranks first. Both come from a routing probe of each run's final checkpoint (step 745–1290). All values are measured.

| Run | Objective | Mass | Cluster gradient | CE gap, step 600 | Distinct experts / token | hitC |
|---|---|---|---|---:|---:|---:|
| 0.3.4 default | mutual information | soft | attached | +0.290 | 3.89 | 0.975 |
| | mutual information | soft | gate only | +0.019 | 6.11 | 0.690 |
| | budget, $B=3$ | soft | gate only | +0.000 | 6.99 | 0.579 |
| 0.3.5 default | budget, $B=3$ | dispatch | gate only | −0.009 | 6.98 | 0.564 |
| $\lambda_{\mathrm{cl}}=0$ rerun | none | – | – | −0.001 | 7.40 | 0.408 |

The $\lambda_{\mathrm{cl}}=0$ reference itself uses 7.45 distinct experts per token. On the trained 0.3.4-default checkpoint, the cluster gradient norm was 0.27 of the CE gradient norm in attention, 0.19 in the gate, 0.11 in the embeddings and 0.07 in routed and shared experts. Detaching the router input removed most of the loss cost, yet the mutual-information objective still cut distinct experts per token to 6.11. The budget objective keeps 6.98–6.99, and its step-600 gap lies within that of the $\lambda_{\mathrm{cl}}=0$ rerun.

### 9.2 Definition

Let $p(\cdot\mid w)$ be the pooled cluster distribution of stream $w$ (sections 1 and 3) and $\bar p$ its weighted marginal. For a budget $B<N_c$,

$$
 L_{\mathrm{budget}}=1-\frac{H(\bar p)}{\log N_c}+\sum_w\alpha_w\,S_B\!\left(p(\cdot\mid w)\right),\qquad
 S_B(p)=\sum_{c\notin\mathrm{top}_B(p)}p_c .
$$

The ranking $\mathrm{top}_B$ is held fixed in the backward pass, so the spill term moves mass into a stream's $B$ heaviest clusters and leaves the split among them free. The loss is zero when the marginal is uniform and every stream stays within $B$ clusters. With $B\ge N_c$ only the balance term remains. `cluster_locality_budget=0` chooses $B=\lceil N_c/2\rceil$. [Torch](../chronos/model/temporal_loss.py#L87), [budget rule](../chronos/expert_clusters.py#L40).

The dispatch mass keeps each dispatch's selected experts:

$$
 \tilde Q_{t,v,e}=\frac{Q_{t,v,e}\,\mathbf1\{e\in\mathrm{top}_k(Q_{t,v})\}}{\sum_{e'\in\mathrm{top}_k(Q_{t,v})}Q_{t,v,e'}} .
$$

With `cluster_input_detach=True`, each training dispatch records one more tensor, $\mathrm{softmax}(W_g\,\mathrm{sg}(x))$, where $\mathrm{sg}$ stops the gradient. Only the cluster term reads it, so this term changes $W_g$ and never $x$. Routing, the CE path and the other auxiliary terms still use the live router. The cost is one extra gate projection per training dispatch. [Torch record](../chronos/model/moe_chronos.py#L160), [MLX twin](../chronos/mlx/training/trainer.py#L358).

A checkpoint saved before 0.3.5 has no `cluster_locality_objective` field. Loading it sets `"mutual_info"`, `"soft"` and `cluster_input_detach=False`, so continued training keeps the objective those weights were trained with unless the caller overrides the fields. [Backfill](../chronos/model/checkpoint.py#L208).

### 9.3 Measurements on a full schedule

These runs use a cosine schedule (peak 6e-3, 2.5 % warmup, floor 0.1) over 2,000 steps, i.e. 131M tokens, with three seeds per arm and the 0.3.5 defaults in segment mode, window 256, $\lambda_{\mathrm{cl}}=0.3$. Valid CE is the mean over 64 fixed validation batches at step 2,000. The resident-memory columns come from a budget probe: per validation window, the $M$ experts selected most often in the first 128 tokens stay resident, any other selection falls back to the shared expert, and tail CE is measured on the remaining tokens. All values are measured with MLX on one Apple M5.

15M shape, 24 experts in 6 clusters, $B=3$; the E12 arm is a free model with 12 routed experts (10.54M total parameters), so all of its experts are always resident.

| Seed | Valid CE free | Valid CE clustered | Δ | Tail CE, E12 free | Tail CE, clustered at 12 resident | Δ | Tail CE, free E24 at 12 resident | hitC free / clustered |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1234 | 4.1489 | 4.1344 | −0.0145 | 3.9443 | 3.9303 | −0.0140 | 3.9890 | 0.438 / 0.586 |
| 2026 | 4.1452 | 4.1509 | +0.0057 | 3.9536 | 3.9319 | −0.0217 | 3.9667 | 0.415 / 0.584 |
| 77 | 4.1560 | 4.1534 | −0.0026 | 3.9447 | 3.9290 | −0.0157 | 3.9861 | 0.424 / 0.596 |

30M shape (hidden size 256, 16 experts in 4 clusters, $B=2$, loop depth 8; 29.68M total and 13.17M active parameters).

| Seed | Valid CE free | Valid CE clustered | Δ | Tail CE at 8 resident, free | Tail CE at 8 resident, clustered | Δ | hitC free / clustered |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1234 | 4.0448 | 4.0258 | −0.0190 | 3.8921 | 3.8287 | −0.0633 | 0.440 / 0.567 |
| 2026 | 4.0284 | 4.0180 | −0.0104 | 3.9081 | 3.8083 | −0.0998 | 0.455 / 0.571 |
| 77 | 4.0199 | 4.0716 | +0.0517 | 3.8842 | 3.8703 | −0.0138 | 0.441 / 0.628 |

At full memory the budget objective is at parity with free routing: the mean Δ is −0.004 at 15M and +0.007 at 30M. Under a resident-memory limit it wins on every seed. At 15M, the clustered 24-expert model with 12 experts resident beats a free 12-expert model by 0.014–0.022, whereas the free 24-expert model paged to 12 loses to it by 0.013–0.045.

Two control runs at 30M show how large run-to-run differences are. They kept $\lambda_{\mathrm{cl}}=0$ for the first 15 % of the schedule and raised it to 0.3 by 30 %. During that phase they compute the same loss as the free runs and differ only by the gate-only record, i.e. a different compiled graph. At step 200 they already differed from the free runs by +0.065 (seed 77) and −0.052 (seed 1234), and seed 77 ended at +0.022. Seed 77's +0.052 is therefore within the spread this setup produces without any cluster loss.

Free routing already specializes by source to a degree. On the 15M probes, the mutual information between a segment's source corpus and its dominant cluster, divided by the source entropy, is 0.43–0.48 free and 0.49–0.52 clustered; at 30M it is 0.25–0.34 free and 0.48–0.51 clustered. The term raises it further; its measured CE benefit appears only under a resident-memory limit. The loop keeps its advantage: on the 15M shape at constant 6e-3 for 800 steps (seed 1234), depth 4 lowers valid CE relative to depth 1 by 0.183 free and by 0.195 clustered.

These measurements do not include the 50M shape, wall-clock paging latency or bytes read from storage. The budget probe models residency by a fixed rule and measures CE, not time. Source: [summary of the run, probe and budget-probe files](../reports/cluster_budget_20261127/summary.json).

**Notation:** CE, cross-entropy; MoE, mixture of experts; MLP, multilayer perceptron; SSD, solid-state drive; RAM, random-access memory; KL, Kullback–Leibler divergence; MLX, Apple's array framework; hitC, cluster-level lookahead hit rate; $\pi$, fixed expert-ID partition; $N_c$, cluster count; $B$, clusters per stream under the budget objective; $V$, collected dispatch count; $\mathrm{sg}$, stop-gradient.
