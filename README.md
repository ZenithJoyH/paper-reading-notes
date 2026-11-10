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
