# DualPace：独立调度的GPU–CPU协同解码

## 论文信息

- 完整标题：DualPace: Decoupled Scheduling and Cooperative GPU–CPU Decoding for LLM Serving。
- 作者：YuXuan Lin、KeQiang Wu、Kingsum Chow。
- arXiv2610.33501v1；2026-10-24 07:10:20 UTC，即北京时间15:10:20。
- [信息页](https://arxiv.org/abs/2610.33501v1)｜[在线全文](https://arxiv.org/html/2610.33501v1)｜[在线PDF](https://arxiv.org/pdf/2610.33501v1)。已在线阅读方法、实现、实验与附录；未下载PDF/TXT。
- 未定位作者专用公开代码，commit/release未披露；不把NEO或vLLM依赖当DualPace实现。
- [本期周报](../../LLM推理研究与工程周报_2026-10-30.md)。以下是作者公开真机结果，无独立复现。

## 研究背景与动机

CPU offload能增加KV驻留容量，但CPU请求是否前进往往受GPU调度窗口支配。即使host已有KV与可用算力，CPU token进展仍可能长时间停滞，造成较差尾ITL。因此，“存在哪里”和“由谁决定何时推进”应分开。

CPU memory bandwidth可处理memory-bound attention，AMX让BF16 dense操作也有实用空间；但GPU仍擅长某些dense batch。需要独立调度而非完全禁止设备间协作。

## 主要贡献

GPU与CPU各自形成batch并推进，无共同iteration barrier；host成为独立decode tier。CPU按batch选择全CPU、仅最终阶段GPU辅助或逐层Fused Assist。profile同时计CPU完成时间和GPU主线被挤占的服务时间，避免只看辅助kernel更快而牺牲总体吞吐。

## 方法论

LLMEngine维护全局请求状态，GPU engine负责prefill及GPU-resident decode；CPU engine负责host-resident decode。控制和activation经固定布局共享内存，避免每stage走Ray object serialization；Ray用于worker生命周期和部署。

CPU的PagedAttention直接读取host HND KV，BF16 dense使用oneDNN/AMX；GPU用FlashInfer attention及CUDA KV管理。GPU TP用NCCL、CPU TP用Gloo，buffer按rank与NUMA划分。

Dense Assist在GPU任务stage边界执行，不抢占活跃kernel。profile分CPU-local compute、shared buffer访问、设备CUDA时间、CPU可见等待、host launch/finalization，再构造route score。CUDA执行同时影响CPU等待和GPU主线容量，两者是不同代价，不把组件堆叠图当严格串行E2E。

阈值是硬件相关的：论文RTX4090示例将CPU batch≤16保留CPU、17—32用Split-Final、更大Fused Assist，GPU token limit1024、CPU decode batch上限256。它们是profile输出，不是对任意异构设备通用的推荐常量。

## 实验条件与结果

| 配置 | 设备／CPU预算 |
| --- | --- |
| Llama3.1-8B | 1×RTX4090，Xeon Silver4510 12物理核 |
| Qwen3-14B | 2×H20，Xeon Platinum8469C 24物理核 |
| Qwen3-32B | 4×H20，同型号48物理核 |

CPU各配额限单NUMA。正文描述BF16 kernel路径，完整weight/activation/KV dtype和软件版本一致性表未逐项给出。baseline vLLM0.22.1、无host KV offload、开chunked prefill、关prefix cache；synthetic token保留trace长度，关闭cache避免重复token造成假命中。Qwen上的NEO有作者兼容patch，缺固定SHA。

Azure Conversation/Code各前2000请求，固定seed指数到达间隔按目标req/s缩放。ISL/OSL图分布未转为完整数表；Code明确平均输出22tokens。客户端并发和完整软件环境未锁定，不补写。

在Conversation、mean(E2E/output tokens)=1s/token目标下，三配置相对NEO可持续req/s分别+43.2%、+6.9%、+18.1%；相对vLLM+90.6%、+11.5%、+23.8%。在P99 ITL1.5s目标下相对NEO为+50.5%、+15.2%、+7.3%。P99 TTFT5s目标附近与NEO接近。请求率用相邻测点线性插值，不外推；不能写成固定req/s下延迟降低相同比例，也不能冒充输出tokens/s提升。

绝对req/s、输出/总tokens/s、逐点TTFT/TPOT/ITL的完整配对数表未提供，本笔记不读图补造。附录TPOT为每请求(E2E−TTFT)/(n_out−1)的P99；ITL是token间隔尾部，二者不等同。

反例：Code的长prompt短输出使prefill主导，在Qwen配置上GPU-only更有利。GPU-only的P99 ITL仍更低，独立CPU调度主要减少NEO的调度性停顿，不能消除CPU执行本身更慢的差距。

host KV120→240GiB吞吐/TTFT基本平台期；更小host池可让更多请求留GPU从而ITL更好，容量不是单调改善ITL的旋钮。CPU TP1→2实验每rank12核，aggregate host KV相同；增加核与内存域带宽本身有成本，不能称同核数纯软件收益。

无assist时stage轮询每层2—5μs，32—64层约0.1ms数量级。Qwen32B/H20与Llama8B/4090的assist host orchestration占组件工作约39.6%/30.0%；CUDA与host可部分重叠。这说明host提交开销需纳入profile，不能只比GEMM。

## 局限性

作者实现未公开、依赖环境缺完整锁定；主要decode trace结果，非多机服务稳定性报告。缺独立质量对照和跨CPU架构验证，AMX外平台需重做kernel及profile。固定profile对负载漂移、NUMA竞争和GPU并行任务的适应性尚需验证。

## 对异构推理测试与优化的启发

建议固定核数、总KV容量、模型精度，比较GPU-gated、CPU独立、CPU独立+assist，再单独扩核；记录host token-progress曲线、GPU主线受扰动时间、P99 ITL与TTFT。4k/1k、16k/1k、客户端并发1/4/64及独立QPS扫描为建议，未执行。

正确性检查共享buffer拥有权、KV迁移完成再交接调度、CPU/GPU数值容差与输出；稳定性覆盖取消、worker退出、buffer复用、NUMA迁移与assist超时。收益预期是资源受压且长decode时提高SLO下服务率，不预设任意国产CPU/NPU都能获得同幅度收益。

## 核心要点

- host驻留容量与独立进展能力必须分别测试。
- 调度独立不等于执行隔离，辅助也会干扰GPU主线。
- 同SLO下req/s、固定负载延迟和token吞吐是不同口径。
- 长prefill短输出有明确负例。
- profile须包含host orchestration和设备扰动，不能仅依赖算子微基准。
