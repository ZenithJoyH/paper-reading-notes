# 配置与缓存架构受控评测：中文论文笔记

## 论文信息

- 完整标题：A Controlled Experimental Evaluation of Configuration and Cache Architecture in LLM Serving Engines。
- 作者：Muhammad Faizan Anwar、Sina Keshvadi；Thompson Rivers University。
- arXiv2611.05991v1，2026-11-04 05:37:14 UTC／北京时间13:37:14。
- [原文](https://arxiv.org/abs/2611.05991v1)｜[HTML全文](https://arxiv.org/html/2611.05991v1)｜[在线PDF](https://arxiv.org/pdf/2611.05991v1)。在线读方法、实验、威胁及artifact附录，未保存PDF/TXT。
- [公开artifact](https://github.com/FaizanAnwar01/llm-serving-config-eval)，tag v1.0指向51bbdfe5ced489d4eaba04e7aceef0ca2e4bc1d0（公开API核对）。本文未执行make或硬件复现。
- [对应周报](../../LLM推理研究与工程周报_2026-11-06.md)。实验是论文作者的第三方框架比较，不是对本论文的独立复现。

## 研究背景与动机

前缀缓存架构常被用来解释框架差距，但默认chunked prefill会随硬件和版本变化，命令行参数也未必等于最终生效值。饱和附近只报TTFT倍率会把小容量差放大成看似架构级数量级差。

论文的核心不是发明新缓存，而是将一个配置差异拿出来干预，并保留未干预的SGLang作为内部控制，分析残余和部署环境变化。

## 主要贡献

可复现GKE testbed、seeded多agent DAG、默认与matched两臂、逐请求日志和claims-checking。将配置贡献、排队放大及未归因残余分别报告；明确不能用一个默认比较证明radix比block cache快几十倍。

## 方法论

单NVIDIA L4 24GB、GKE1.35、独立CPU节点generator，固定单副本和模型本地volume。vLLM0.6.3 V0和SGLang0.4.1-cu121均服务Llama3.1-8B-Instruct BF16、context16384、prefix cache开。默认vLLM不chunk、SGLang2048；matched给vLLM开chunk并设置max-num-batched-tokens2048，其他差异仍存在。

vLLM内存参数0.88、SGLang0.85的语义不等同；attention backend、waiting顺序及cache granularity也未匹配。论文从启动日志记录resolved config，这是最值得复制到异构测试的方法。

每个DAG=planner+4workers+aggregator，6个请求带依赖和tool delay；Gamma graph到达CV1.5，graph/s不是request/s。共享前缀与unique suffix精确校验token数及SHA，输出planner/workers上限128、aggregator256。固定seed配对3次，rate sweep主要输入4096/sharing0.9、0.05—0.40graphs/s、600/900s；另sharing和长度扫描，总99runs/63,288请求。

Warmup20请求及首graph排除；server metrics5s采样，按run window取数。Spot换节点时整条件三次重做。两个campaign独立provision；SGLang同配置也快0.5—11.7%，表明环境并非完全相同。

## 实验结果

### 作者公开真机

0.28graphs/s（名义1.68req/s）处，默认吞吐比SGLang/vLLM1.22，matched1.046；论文的“移除78%差距”不是吞吐提高78%。同点TTFT P50比36.4→12.3，vLLM TPOT112→125ms。TTFT改善与decode节奏代价并存。

matched实际完成vLLM1.58、SGLang1.66req/s，vLLM不能跟上，队列继续积累。低一些的0.12graphs/s两者均跟上流量，matched TTFT p95仍相差2.44×；8192输入下p95差6.43×，但该点两者都只完成约89%offered load。不能概括所有残余比较都无排队。

原文不报P99：最轻负载仅不足200请求，不支持可靠估计。主要指标为TTFT p50/p95、TPOT median、完成req/s，不是output tokens/s。sharing扫描只在默认配置下做，不能证明matched时同样反转。

### 分析模型

流体queue近似用吞吐shortfall预测等待时间随窗口T增长：中位等待约delta/(1−delta)×T/2。真机相关性r0.85，但平均高估1.6倍；DAG依赖会反向拖慢请求发出，使系统并非纯open-loop。因此这是解释模型，不是另一套硬件实测。

### 未被证明的归因

剩余尾差与SGLang最长prefix优先相容，但没有固定排序、attention kernels和真实KV容量的独立消融。不能称已证明由LPM、radix或block granularity任一因素引起。

## 局限性

旧版本、单模型单卡、固定context及synthetic tokens，不外推当前vLLM/SGLang或多机EP。三个重复置信区间宽，部分多重比较未作统一校正；部署与配置变化混杂。运行时未记录image digest，附录digest为事后tag页面值；累积hit counter混合多run，不能当每run命中率。

repo声称包含99run日志、Terraform、generator与claims-check，README给make/verify命令；本次仅检查可访问性与tag，不声称已验证全部107条claims。复现需GKE quota、L4、gated模型license/token和固定Python依赖。

## 对异构推理测试优化的启发

建议先固定真实KV pool容量、chunk/token budget、backend及prefix策略，再比较平台；从低负载扫到饱和，同时报告offered/achieved rate、失败和queue。4k/1k、16k/1k、并发1/4/64与独立QPS扫描是建议，不是论文统一已有实测。

正确性先检查tokenization、prefix hash、输出/EOS、stream chunk与token映射；稳定性检查client发送lag、节点替换、warmup、冷/热cache和counter窗口差分。不要为凑P99用太少样本，不能将max-num-seqs当客户端并发。

## 核心要点

- 测实际生效配置，不只记录启动命令。
- 小吞吐短缺可随运行时长累积成大TTFT比。
- TTFT改善可能伴随TPOT变差。
- matched一个设置不等于消除所有混杂。
- 该论文贡献是评测方法，不是当前框架排行榜。
