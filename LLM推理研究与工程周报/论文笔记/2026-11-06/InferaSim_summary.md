# ROCm InferaSim：少量整步测量驱动部署筛选

## 论文信息

- 完整标题：ROCm InferaSim: Cut LLM Serving Trial Costs and Deploy the Best Configuration。
- AMD Anshu Raina、Yao Fu、Zhang Jiejing、Peyman Razaghi等14位作者。
- arXiv2611.04032v1，2026-11-02 22:23:40 UTC／北京时间11-03 06:23:40。
- [论文](https://arxiv.org/abs/2611.04032v1)｜[全文](https://arxiv.org/html/2611.04032v1)｜[在线PDF](https://arxiv.org/pdf/2611.04032v1)。已在线读设计、评测、局限与复现附录；无PDF/TXT归档。
- 论文说明anchors/corpora/scripts将在camera-ready公开，目前无核验到专用公开artifact/release。依赖RCCL、ATOM、vLLM不等于预测器代码。
- [本期周报](../../LLM推理研究与工程周报_2026-11-06.md)。以下分作者校准实测和预测结果；本文未复现。

## 研究背景与动机

推理性能取决于engine、kernel、TP/PP/EP、精度、shape、调度、spec及KV层级。每个候选都跑真机成本高，纯理论模型又容易漏固定host和launch成本，逐算子数据库也未必对应实际engine融合路径。

论文用整步测量覆盖实际graph、padding和host开销，再通过解析传递和事件模拟筛配置，目标是在有限GPU预算下选值得真机确认的Top-K。

## 主要贡献

anchor整步标定；多节点reduce-measure-restore；通信/PP/EP/host tier和spec模型；固定与agentic负载的校验；同engine内排序与部署mix筛选。没有证明模型可以取代任务准确率、硬件故障和多机长期稳定性实验。

## 方法论

Anchor绑定accelerator、engine build、precision、kernel/layout，保留固定每步成本，只缩放可分摊工作。覆盖目标一般须同build/layout且并发在4倍内；DCP改变attention kernel需独立anchor。无anchor的simulate模式只适于排序，尤其同engine候选的共享偏差可抵消，不宜跨engine绝对计时。

离散事件模拟跟踪queue、batch、前缀、层级KV、P/D pool及transfer；PP模型加每stage无法隐藏的host与transfer成本。MoE考虑最忙rank，不能只用平均expert负载。spec接受率需要建模；MTP有测量，EAGLE/DSpark/ngram部分仅预测，不能统一称实测。

调参器先固定seed planner逐项改lever，再可用LLM提出候选，所有候选由同一预测与合法性检查计价。论文192次大规模搜索只执行deterministic seed阶段、无LLM调用；不能据其宣称LLM搜索已被充分验证。

## 实验条件与结果

### 作者校准／验证真机

节点8×MI355X，每卡288GB HBM3E、Infinity Fabric；ROCm7、graph capture开。跨机Pollara RoCE、DMA-BUF、PXN。Engine为vLLM0.30.0及29468dde、SGLang0.5.19-rocm720-mi35x的日期镜像、ATOM两个dated nightlies；部分nightly已不托管，完整复现仍需补image digest。

固定负载多模型：1024/8192输入、1024输出、并发1—256、TP2—8；每点10C prompts、random tokens、ignore_eos。模型/精度按anchor逐项匹配，不存在一个可泛化所有结果的统一dtype。

11anchors花9.1 GPU-hours，覆盖226配置直接测量花204 GPU-hours；评分另过滤为172配置。不能将226都算进172的误差统计。TPOT/throughput/TTFT median绝对百分比误差8.1%/8.8%/13.6%；93%的TPOT点在20%内。TP8由TP4预测吞吐误差9.7%，anchor原宽度为6.1%。

88个请求数≥160的runs中，P99 TPOT/E2E/ITL median误差10.7%/9.8%/13.8%；重复run自身这些P99差1—4%。这是预测误差，不是指标改善。

### Agentic预测校验

MI355X上AgentX/AIPerf、184配置、1—384sessions，28anchors；其中95 covered held-out的throughput/ITL/TTFT error factors1.08/1.14/1.23，为几何误差因子，不是MAPE。SGLang+HiCache子集1.15/1.40/1.38，说明命中/淘汰时序对时延预测更难。

159个额外InferenceX P/D fixed-length部署，输入1k或8k、输出1k、TP4/TP8 pools、部分attention DP，无P/D anchor；throughput/interactivity/TTFT median误差5.5%/5.3%/17%。这不证明跨机故障/抖动都能预测。

### 排序与筛选

fixed decisive pairs中anchored吞吐排序正确93.4%、TPOT85.4%；agentic同engine比较为90.9%吞吐，51个ITL pairs全对。23个可行model-tier slots，Top5包含实测最优21个，22个可在最优12%内；不是每个slot都必然在12%内。

预测169,962点是CPU投影空间，不是这么多真机测量。12—40GPU P/D布局同理。单次projection0.17—0.48s不含初始采集成本，不能称完全免GPU获得同等精度。

## 局限性

仅MI355X验证，NVIDIA和国产设备仍需采anchor、通信校准及独立holdout。无公开artifact阻碍直接复现；某些engine images不再托管。TTFT、host-tier命中与prefix策略误差更大。Fusion节省是定价模型，不代表每个建议kernel都已实现并实测。

## 对异构推理测试优化的启发

建议先按本机engine/kernel/layout定义anchor signature，再在同engine候选中筛Top3—5；每次版本或graph/backend改变要重新评估anchor有效性。4k/1k、16k/1k、低并发与64并发为建议，不是该论文统一已有实测。

验证排序比只看median误差更重要；同时报告tail误差、错误选择代价、资源总数与P/D双池计数。留出改变并发、context及host tier的真实holdout，不能用同一数据调参再宣称泛化。

正确性单独做量化/采样/长文本准确率，稳定性单独做通信故障、host spill、图重捕获与长时运行。若预测选择违反真实容量或SLO，应回退已验证配置，不能靠LLM解释覆盖失败测量。

## 核心要点

- 整步anchor包含实际执行路径，但只对兼容kernel集合有效。
- 真实采集成本与预测成本分开统计。
- 排序可靠不等于绝对TTFT可靠。
- HiCache、DCP和MoE热点要独立校验。
- 模拟用于缩小真机搜索，不代替正确性与长稳实验。
