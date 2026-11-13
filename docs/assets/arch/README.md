# Chronos 可配置架构图

四幅准确机制图均由 `make_arch.py` 生成。它们使用符号化规模参数，保留 MLA/SWA 与 Dense/MoE 的四种配置组合。输入 token embedding 与输出线性 LM head 分开绘制。四种组合是配置替代项，不表示固定 ABCE 顺序。

| 文件 | 内容 |
|---|---|
| `chronos_forward_stack.*` | 前向堆栈、循环控制与嵌入的块内部电路 |
| `chronos_block_interior.*` | 四种块配置及残差连接 |
| `chronos_ssd_kernel.*` | 专家布局、Torch/CUDA 数据通路、冷缺页策略与 MLX 边界 |
| `chronos_expert_communities.*` | 路由局部性目标、固定分区与离线共激活聚类 |
| `captions.md` / `source_map.md` | 图注、缩写、实现范围及源码锚点 |

普通路径中的每个物理块均有 attention 和 gated FFN 两个预归一化残差子层。默认 SiLU 对应 SwiGLU；Torch 的激活函数可配置。共享专家存在时，其均值分支与 routed 分支并联。`first_only` 的后续迭代跳过 attention。

分页图的主数据通路明确限定为 Torch/CUDA。Torch 可在 `get_tensor` 前筛选页。MLX 的读取器可能先暂态加载并求值整个聚落，再限制 warm-cache 接纳；有界驻留不等于有界暂态加载。详见 [内核章节](../../KERNEL_ARCHITECTURE.md) 和 [懒加载章节](../../LAZY_EXPERTS.md)。

冷 `on_demand` 缺页可排队并使用 shared fallback。`sync_on_demand` 等待准确专家。RAM 热页的提升需要启用 demand loader；`fallback_diagnostic` 不安装该 loader。fallback 不构成对 full-resident 输出的等价保证。

```bash
python3 make_arch.py --outdir ./rendered --formats png svg pdf --dpi 600
python3 make_arch.py --config architecture_config.json --outdir ./parameterized
```

需要 Python、Matplotlib 与 NumPy；不需要加载模型或联网。生成的 PNG、SVG、PDF 与文字转路径 SVG 可用于阅读或继续编辑。生成器在导出前检查文字边界、重叠和连线。可用 `--layout-report /path/to/report.json` 另外保存检查结果。
