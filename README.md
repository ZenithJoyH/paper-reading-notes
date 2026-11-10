# 论文阅读笔记

记录论文与技术文章的研究问题、核心方法、实验结果与适用边界，并通过具体例子解释算法原理和实现细节。

## 持续阅读与更新方式

每篇论文或文章的总结 Markdown 随阅读和对话持续完善。首次总结之后，围绕该文的追问、推导、例子、实现细节、方案比较及纠错，都应同步整合到原有总结中，无需另行要求更新。新增内容融入相关章节，保留已有有效内容，并区分原文结论、补充解释与尚待验证的推断。

除非明确要求仅讨论或暂不写入，agent 应在回答的同时完成笔记更新，并给出更新文件的链接。具体执行约定见 [AGENTS.md](AGENTS.md)。

## 阅读目录

| 论文 | 阅读笔记 | 原文 |
| --- | --- | --- |
| RadiK: Scalable and Optimized GPU-Parallel Radix Top-K Selection | [中文解读](RadiK_Scalable_and_Optimized_GPU-Parallel_Radix_Top-K_Selection_summary.md) | [arXiv 2501.14336](https://arxiv.org/abs/2501.14336) |
| LoRA: Low-Rank Adaptation of Large Language Models | [中文解读](LoRA_Low-Rank_Adaptation_of_Large_Language_Models_summary.md) | [arXiv 2106.09685v2](https://arxiv.org/abs/2106.09685v2) |
| Push-Based KV Transfer to Improve TTFT and TPOT for Disaggregated Inference in vLLM | [中文工程解读](Push-Based_KV_Transfer_to_Improve_TTFT_and_TPOT_for_Disaggregated_Inference_in_vLLM_summary.md) | [vLLM 官方博客](https://vllm.ai/blog/2026-10-14-nixl-push-connector) |
| Super-agent 设计探索：迈向更自主的 Agent 系统 | [中文工程解读](Super-agent_设计探索_迈向更自主的_Agent_系统_summary.md) | [DeepSeek 官方文章](https://zhuanlan.zhihu.com/p/2099987320443237060) |

## 历史对话回补核对（2026-11-09）

已检查项目内 13 份总结，并对照可读取的“LoRA论文解读”“解释 LoRA 是什么”“Radix Select论文解读”“解读知乎文章”“分析知乎文章”“筛选LLM推理工程新进展”共 6 个相关聊天的 30 个回合，包括已归档聊天。本次核对关注讨论内容是否落入笔记，不代表重新验证全部历史实验或覆盖不可读取的聊天。

| 对应笔记 | 核对结果 |
| --- | --- |
| [LoRA](LoRA_Low-Rank_Adaptation_of_Large_Language_Models_summary.md) | 补入秩的数值例子、右下角 6／7 的约束、提高秩后的形状与参数代价、LoRA 与后训练及 SFT／DPO／强化学习的 SQL 教学对照 |
| [Super-agent](Super-agent_设计探索_迈向更自主的_Agent_系统_summary.md) | 补入常驻 Router、多个渠道与订阅直通、任务／Thread／Session 的关系、Router 三层上下文与未披露的压缩策略；另补已讨论的原型授权和状态管理细节 |
| [Push-Based KV Transfer](Push-Based_KV_Transfer_to_Improve_TTFT_and_TPOT_for_Disaggregated_Inference_in_vLLM_summary.md) | 补全 QPS、客户端并发、max-num-seqs 的区别及 8 请求／秒×2 秒≈16 个在途请求的稳态例子 |
| [RadiK](RadiK_Scalable_and_Optimized_GPU-Parallel_Radix_Top-K_Selection_summary.md) | 追问已收录：局部排名与重复值、24→7 次全局原子更新、10→2 次位置申请、两倍缓冲容量、共享内存竞争与硬件边界；未重复添加 |
| [分层 KV 缓存性能建模](LLM推理研究与工程周报/原文/2026-10-14/2610.15366v1_summary.md)、[KV-Repurpose](LLM推理研究与工程周报/原文/2026-10-16/2610.13060v1_summary.md)、[BLIS](LLM推理研究与工程周报/原文/2026-10-16/2610.14255v1_summary.md)、[MultiAlloc](LLM推理研究与工程周报/原文/2026-10-16/2610.18642v1_summary.md) | 已检查，未在上述可读取聊天中发现这些论文另有未入档的追问解释 |
| [HAWK](LLM推理研究与工程周报/论文笔记/2026-10-23/HAWK_summary.md)、[KVCompass](LLM推理研究与工程周报/论文笔记/2026-10-23/KVCompass_summary.md)、[DualPace](LLM推理研究与工程周报/论文笔记/2026-10-30/DualPace_summary.md)、[配置与缓存评测](LLM推理研究与工程周报/论文笔记/2026-11-06/ConfigCacheEval_summary.md)、[InferaSim](LLM推理研究与工程周报/论文笔记/2026-11-06/InferaSim_summary.md) | 已检查，相关聊天只有周报生成与结果汇总，未发现额外追问遗漏 |

另将“自适应验证究竟省什么、为何接受长度下降仍可加速”的流程和 3.4／10ms 与 3.1／7ms 教学例子补入 [2026-10-16 周报第 3 节](LLM推理研究与工程周报/LLM推理研究与工程周报_2026-10-16.md)。该文章已有周报解读，沿用原位置；教学计算与作者实测分别标注。

## RadiK 笔记内容

- 基数选择如何逐轮定位第 k 大的边界，再筛出 top-k。
- 分层原子操作如何减少全局更新与跨线程块竞争。
- 聚合写回如何一次申请连续输出空间，以及缓冲区容量的设计。
- 共享内存原子操作的访问成本、竞争范围和硬件差异。
- 批量任务调度、地址对齐、自适应缩放与性能实验。

笔记中的教学例子用于说明机制，与论文实测数据分别标注。论文原文的权利归其作者及相应权利人所有。

## 工程周报

- [LLM推理研究与工程周报与论文笔记](LLM推理研究与工程周报/README.md)
- [国产AI加速器推理生态周报与历史期次](国产加速器/README.md)

## 工程专题

- [vLLM 推理计算与通信库清单](vLLM_推理计算与通信库清单.md)：Attention、GEMM 与量化、MoE、通信、KV Cache 及底层编译工具，核对日期 2026-10-20。
