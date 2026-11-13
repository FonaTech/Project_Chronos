# Chronos 绘图脚本重构与可读性改进

重构了原脚本的 22 个绘图入口；现有数据支持的 18 张图已生成。另四张需要尚未提供的 benchmark JSON，已在 manifest 中明确列出。原脚本、原始项目和实验数据未被覆盖。

## 使用

```bash
python3 -m pip install -r requirements.txt
python3 make_figures.py
python3 make_figures.py --repo /private/tmp/chronos_release_033 --outdir ./rendered
python3 make_figures.py --figures loop_depth_ce_curves per_source_ce --formats pdf svg --dpi 600
python3 make_figures.py --list
```

运行时请将 `make_figures.py` 与 `chronos_diagrams.py` 放在同一目录。默认使用包内 `source_data/` 快照，便于复现；`--repo` 可改用真实项目目录。所有输出写入 `--outdir`，默认是脚本旁的 `figures/`。不加载模型，不运行训练，不联网。

## 输出

- `figures/*.png`：600 dpi；按 180 mm 双栏宽度设计。
- `figures/*.pdf`：可直接放入论文，嵌入字体。
- `figures/*.svg`：保留可编辑文字；字体为 DejaVu Sans。
- `figures/*_outlined.svg`：文字转路径，无需收件方安装字体。
- `figures/data/*.json`：每条绘图序列、原始文件哈希、字段、计算变换和测量范围。
- `figures/figure_manifest.json`：生成状态、缺失数据和排版审计。
- `figure_overview.png`：缩略总览；读数以独立 PDF / 原尺寸图片为准。
- `all_figures.pdf`：全部已生成图片的矢量 PDF 合集。
- `data_verification.json`：绘图数值与原始记录的独立核对。

## 可读性调整

统一字号、颜色、线宽、淡网格和面板字母。最低字号为 6 pt，定量图大部分标签为 7–9 pt；实际使用时建议保持双栏宽度。颜色、线型和点形共同区分配置，灰度打印仍可辨认。

图例移出数据区域。标题缩短并与实验条件分层。放大图按实际显示的数据确定纵轴范围，避免固定范围裁掉曲线。柱形图保留零基线，并直接标注数值；窄范围 CE 比较改用点图，避免截断柱形夸大差异。

分语料结果改成标数值的热图，并列出每组完成步数。推测解码的箱线图同时显示全部提示词样本和配对连线；箱体为四分位范围，中线为中位数，须线为常规 1.5 IQR 范围。点的水平偏移只用于防遮挡，不改变纵向测量值。未生成独立重复实验的误差条或置信区间。

## 数据与科学含义核对

未平滑曲线、外推缺失点或补造实验样本。相同步数比较沿用最近的原始 evaluation；配对 CE 差仅使用两个实验真实共有的步数。Sigmoid 步长从各 arm 的实际配置边界和原始 raw dt 计算，不再写死层数和边界。

修正了几处会影响科学含义的原始标签：

1. Residency 图的 `ssd_loads_per_token` 是同步需求加载计数，不是 cluster 文件读取次数。
2. Locality sweep 的 reads 从 JSON 读取精确记录，不再使用代码中写死的近似列表；CE 差仍沿用原图的最后一次 curve CE 定义，并明确标为 final recorded CE。
3. 长上下文图读取完整 JSON，正确区分整块重计算和 MLP 重计算，保留两者各自存在的数据点。空缺表示未运行，不表示零。上下文长度按 log₂ 间距显示。
4. `peak_gb` 没有在这些 JSON 内附带单位换算说明，相关图标为 reported GB，未擅自推断为 GiB。
5. 单层内存图的 MLA 数值可追溯到 `docs/ATTENTION.md` 表格；SWA 数值仅见原绘图脚本，未提供原始测量文件。因此整图标为 reported/unverified，不冒充原始实测。保留 reported OOM，不将缺失值画成零，也未保留未经支持的 >64 下界。
6. 架构示意沿用上一轮已按实现核对的模块图，输入和输出 embedding 分开绘制。Lookahead 预测未来 token 的专家概率，不画成必然的“下一层聚落预测”，预取路径也不直接合入当前激活输出。
7. `kernel_evolution` 保留文件名以兼容原引用，内容改为源码支持的执行配置比较，不继续给版本绑定未经验证的历史架构。版本日期从 CHANGELOG 自动读取，未把原脚本中未有记录的 0.3.3 稳定版日期当成事实。

这些名称或范围的修正已在数据记录中说明；原始测量数值保留在快照中。

## 缺失的可选数据

以下四个文件在用户给出的项目目录中也不存在，四个绘图函数已经完成样式重构，但没有生成虚构预览：

- `experiments/figures/benches/out/moe_balance.json`
- `experiments/figures/benches/out/router_entropy.json`
- `experiments/figures/benches/out/lazy_inference_trace.json`
- `experiments/figures/benches/out/prefetch_depth.json`

数据补齐后重新运行即可。`--strict` 可让缺失数据导致非零退出码，适合 CI；默认会生成所有可用图并记录缺项。排版失败始终会报错。

## 审计与修改

`set_style()` 控制字体和色彩；`canvas()` 控制物理尺寸、面板和留白；`LABEL/COLOR/MARKER/LINE` 保证跨图一致性；各 `fig_*` 函数处理对应图。`chronos_diagrams.py` 管理架构图的毫米坐标和正交连接。

脚本检查画布文字溢出、相邻刻度文字重叠、图例覆盖数据区域；架构图还检查节点标签、文字重叠和连线穿框。所有定量序列写入 JSON 以便独立数值核验。输出是原始记录的重新可视化，没有在本次工作中重跑性能实验。

## 数值复核

完整重建后运行 `python3 verify_values.py`，将导出的序列与 source_data 中的记录逐项比较。本次完成 87 项检查、880 个数值条目核对；除 sigmoid 逆变换使用 1e-12 浮点容差外，其余数值检查为精确一致。该验证未重跑测量实验。

缩写：CE，交叉熵；MLA，多头潜在注意力；SWA，滑动窗口注意力；MLP，多层感知机；MoE，混合专家；KV，键值；LRU，最近最少使用；SSD，固态硬盘；RAM，随机存取存储器；IQR，四分位距。

快照中的 `docs/ATTENTION.md` 仅用于其内存数据表。该文档含有与实现不一致的历史架构说明，架构图仍以已核对的实现为准。
