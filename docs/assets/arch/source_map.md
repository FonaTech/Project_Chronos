# 图中机制与源码对应表

下表将架构图中的模块与仓库源码相对应。图中符号表示可配置参数，具体执行路径由解析后的模型配置决定。

- **block** — Every ChronosBlock contains attention and an MLP, with two pre-norm residual sublayers. [ChronosBlock](../../../chronos/model/model_chronos.py#L48)
- **parity** — Hybrid attention uses MLA on even physical layers and SWA on odd physical layers; hybrid-off uses global GQA. [make_attention](../../../chronos/model/hybrid_attention.py#L405)
- **embedding** — Input token lookup and final RMSNorm are backbone modules. [ChronosModel.__init__](../../../chronos/model/model_chronos.py#L135)
- **head** — Output is a vocabulary Linear LM head; tying to input embeddings is optional. [ChronosForCausalLM.__init__](../../../chronos/model/model_chronos.py#L551)
- **execution** — Prelude and coda are normal blocks. Block-order or layer-order recurrence applies only to the recurrent physical groups. Lookahead observes the post-prelude state. [ChronosModel._forward_recurrent](../../../chronos/model/model_chronos.py#L204)
- **conditioning** — Loop and group embeddings condition each recurrent call. [RecurrentController.condition](../../../chronos/model/recurrent.py#L136)
- **Euler** — The contracted update subtracts the unconditioned before state, not the conditioned state. [RecurrentController.contract](../../../chronos/model/recurrent.py#L147)
- **moe** — FP32 softmax, top-k routing, optional probability normalization and a MEAN of shared experts when configured. Runtime cold fallback is policy-dependent. [ChronosMOEFeedForward](../../../chronos/model/moe_chronos.py#L36)
- **training_partition** — Training clusters are fixed contiguous expert-ID groups, not an emergent spatial clustering of parameter vectors. [training_clusters](../../../chronos/expert_clusters.py#L66)
- **auto_paging** — Auto paging uses training clusters only when cluster-locality training is enabled; otherwise singleton expert pages. [inference_clusters](../../../chronos/expert_clusters.py#L87)
- **locality_objective** — The optional locality objective concentrates per-stream routing and preserves diversity across streams. [cluster_locality_loss](../../../chronos/model/temporal_loss.py#L87)
- **segment_objective** — Segment mode pools document-level dispatch probabilities; it is an alternative to fixed windows. [segment_cluster_loss](../../../chronos/model/temporal_loss.py#L119)
- **offline_graph** — Offline coactivation graph can be partitioned by Louvain, with greedy fallback. This is distinct from the training partition. [cluster_experts_louvain_or_greedy](../../../chronos/io/cluster_layout.py#L219)
- **selective_pages** — Torch filters requested layer/expert pages before materialization from a cluster file. MLX can transiently evaluate the whole cluster before bounded admission. [ClusterStorage.load_cluster](../../../chronos/io/storage.py#L169)
- **bounded_prefetch** — Cluster neighbours are admitted under a RAM capacity bound; the requested page is protected by ordering. [ExpertStore._prefetch_clustered](../../../chronos/io/expert_store.py#L560)
- **async** — A bounded queue drives background SSD-to-RAM loading; predictions and weight transfers are distinct. [AsyncPrefetcher](../../../chronos/io/async_prefetcher.py#L29)
- **events** — Compute waits for the necessary expert events after promotion. [CacheManager.ensure_resident](../../../chronos/runtime/cache_manager.py#L117)
- **miss_policy** — Cold on_demand queues an exact request and falls back now; sync_on_demand waits to load the exact expert. [ChronosInferenceEngine._install_moe_runtime_hooks](../../../chronos/runtime/inference_engine.py#L121)
- **mlx** — MLX prefetch produces evaluated arrays in unified memory, rather than a CUDA H2D copy. [MLXExpertStore](../../../chronos/mlx/expert_store.py#L61)
- **next_use** — Next-use residency planning is implemented as an optional MLX runtime policy. [NextUsePlanner](../../../chronos/residency_planner.py#L40)

