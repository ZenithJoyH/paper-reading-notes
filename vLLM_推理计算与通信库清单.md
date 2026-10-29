# vLLM 推理计算与通信库清单

核对日期：2026-10-20。依据 vLLM 官方 `latest` 文档和 `main` 源码；以 NVIDIA GPU 为重点，补充 AMD ROCm。主线已集成不代表用户所用发行版已包含，也不代表一次推理会调用全部组件。本文整理主要计算与通信生态，不是完整 Python 依赖树；未检查用户的实际部署环境。

## 1. 先区分库、内核、量化方案

- **独立库／项目**：DeepGEMM、FlashInfer、FlashAttention、FlashMLA、DeepEP 等。
- **编程与编译工具**：Triton、CUTLASS／CuTe、CuTe DSL、TileLang 等，用来实现或生成计算内核。
- **vLLM 内集成的内核实现**：Marlin、Machete、ExLlama 路径、融合 MoE、自定义 AllReduce 等；不一定以独立 pip 包出现。
- **量化算法与数据格式**：AWQ、GPTQ、FP8、NVFP4、MXFP4 等，决定权重和激活如何表示；执行计算还需要选择合适的内核。
- **量化工具**：LLM Compressor、Model Optimizer、Quark 等，主要用于准备模型。模型由哪个工具量化，与推理时调用哪个 GEMM 库，是两个问题。

官方量化说明将格式、工具和执行支持放在同一生态中介绍；具体执行实现可以从 [Linear 内核导出源码](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/kernels/linear/__init__.py)和[量化文档](https://docs.vllm.ai/en/latest/features/quantization/)交叉核对。

## 2. Attention：计算注意力

| 库／实现 | 负责什么 | 常见使用条件与边界 |
| --- | --- | --- |
| FlashAttention（FA2／FA3／FA4） | 融合注意力计算，减少中间结果读写 | 标准 MHA／GQA／MQA，也有 MLA 集成路径；版本取决于 GPU、形状和功能需求 |
| FlashInfer | 面向推理的 Attention、Paged KV 访问等；另有 GEMM、MoE 实现 | 覆盖 prefill、decode；部分路径通过其接口使用 TRT-LLM 来源的内核 |
| FlashMLA | 针对 Multi-head Latent Attention 的高效内核 | DeepSeek 风格 MLA，包含 dense／sparse 相关后端；不能据此认为普通 GQA 模型也使用它 |
| TokenSpeed MLA | MLA 加速实现 | 当前 CUDA 依赖和 MLA 后端表已集成；GPU 与模型维度有约束，较旧发行版未必包含 |
| Triton Attention／FlexAttention | Triton 实现或基于 PyTorch FlexAttention 的注意力路径 | 由兼容性与功能需求选择；它们不是 FlashAttention 的别名 |
| CUTLASS MLA | 基于 CUTLASS 的 MLA 实现 | 当前有针对 Blackwell 的后端；CUTLASS 本身还用于其他算子 |
| AITER Attention | AMD GPU 上的注意力实现 | ROCm 平台，官方列出 `ROCM_AITER_FA`、`ROCM_AITER_UNIFIED_ATTN` 等后端 |

Attention 后端由模型数据类型、KV Cache 格式、head size、GPU 架构、mask 和并行功能等共同决定。当前文档的标准 Attention 自动选择顺序在 Ampere／Hopper 上优先 FlashAttention，在 SM10.x Blackwell 上优先 FlashInfer，但特殊配置和版本可能改变结果。MLA 的 prefill 与 decode 还可分别选择实现。

来源：[Attention 后端与功能表](https://docs.vllm.ai/en/latest/design/attention_backends/)、[CUDA 依赖清单](https://github.com/vllm-project/vllm/blob/main/requirements/cuda.txt)。

## 3. GEMM、量化 Linear 与专家计算

GEMM 即通用矩阵乘。Transformer 的 QKV 投影、输出投影、MLP 和 MoE 专家中的线性层，都需要矩阵乘；它们不一定采用同一个后端。

| 库／内核 | 主要用途 | 在 vLLM 中应如何理解 |
| --- | --- | --- |
| **DeepGEMM** | 低精度 GEMM、Grouped GEMM 等 | FP8 Linear／MoE 专家计算的重要实现；上游现已扩展到 FP4、BF16、Indexer、Mega MoE 等，但具体启用范围必须看 vLLM 集成 |
| **CUTLASS／CuTe** | NVIDIA Tensor Core 内核模板与布局抽象 | 构建 FP8／FP4 GEMM、MoE 等内核的基础；常被编译进扩展，并非运行时单独启动一个 CUTLASS 服务 |
| **Marlin** | 低比特权重的融合反量化与矩阵乘 | 常见于 GPTQ／AWQ 等权重压缩模型，也有 FP8／FP4 扩展；vLLM 内同时有 Linear 和 MoE 路径 |
| **Machete** | 混合精度、低比特权重 GEMM | 当前构建脚本仍集成，面向 Hopper／SM90a；不是所有 GPU 都可用的通用依赖 |
| **Humming** | 量化 GEMM 内核 | 当前 CUDA 主线依赖包含 `humming-kernels`；Linear 注册中有混合精度、MXFP4／MXFP8、NVFP4 等实现 |
| **FlashInfer GEMM／MoE** | 推理矩阵乘与融合专家计算 | FlashInfer 不仅提供 Attention；可暴露 CUTLASS、CuTe DSL、TRT-LLM 等计算路径 |
| **Triton／triton_kernels** | GPU 算子编写、编译与内核集合 | 支撑融合 MoE、量化和其他算子；`triton` 编译工具与 `triton_kernels` 内核集合应分别理解 |
| **ExLlama 内核路径** | 量化 Linear 的执行实现 | 当前 Linear 导出源码仍有 `ExllamaLinearKernel`；应结合量化格式和版本查看是否实际选中 |
| **bitsandbytes** | 低比特加载与计算 | vLLM 支持的可选量化集成，涉及 4-bit／8-bit；不意味着所有量化模型都通过它执行 |
| **AITER** | AMD 上的 GEMM、融合 MoE 和其他算子 | ROCm 的重要优化库；不能把 NVIDIA 的 DeepGEMM／Marlin 支持直接套到 AMD |

来源：[融合 MoE 功能表](https://docs.vllm.ai/en/latest/design/moe_kernel_features/)、[Linear 内核源码](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/kernels/linear/__init__.py)、[构建脚本](https://github.com/vllm-project/vllm/blob/main/CMakeLists.txt)、[DeepGEMM 上游](https://github.com/deepseek-ai/DeepGEMM)、[量化文档](https://docs.vllm.ai/en/latest/features/quantization/)。

特别注意：当前 vLLM 的 [DeepGEMM 构建配置](https://github.com/vllm-project/vllm/blob/main/cmake/external_projects/deepgemm.cmake)使用固定提交的 `vllm-project/DeepGEMM`，并打包到 `vllm/third_party/deep_gemm/`。因此，不能用 `pip list` 是否出现独立 DeepGEMM 包来判定是否启用，也不能把上游最新版的全部能力视为当前 vLLM 的能力。

## 4. 内核编程与基础运行环境

| 组件 | 作用 | 与上述组件的关系 |
| --- | --- | --- |
| PyTorch | 张量、模型执行、算子调用、编译与分布式基础 | vLLM 的基础运行框架，部分算子进入 PyTorch 实现，部分调用自定义扩展 |
| CUDA 及 cuBLAS／cuBLASLt | NVIDIA 运行环境与矩阵计算基础 | 普通 PyTorch CUDA 矩阵运算可能调用这些基础库；特化量化内核则走相应实现 |
| Triton | Python 风格 GPU 内核语言与编译器 | vLLM 自有算子、融合 MoE 和编译生成代码的重要基础；与 NVIDIA Triton Inference Server 是不同项目 |
| CUTLASS／CuTe 与 CuTe DSL | Tensor Core 内核构建工具 | CUTLASS C++ 模板／CuTe 布局抽象与 Python CuTe DSL 有关联，但应区分；当前 FA4 路径引入 `nvidia-cutlass-dsl` |
| TileLang／TVM FFI | GPU 内核编程与调用基础 | 当前 CUDA 和 ROCm 依赖都包含 TileLang；不能据此认定每个模型都会执行 TileLang 算子 |
| cuDNN Frontend | NVIDIA 算子与执行计划相关接口 | 当前 CUDA 依赖包含，具体使用取决于后端路径；不是 DeepGEMM 的同义替代品 |

当前主线还出现 `quack-kernels`，CUDA 依赖注释说明其用于 MSA。它属于特定模型／算子路径，可作为扩展项跟踪，不应列为所有推理请求必经组件。

来源：[CUDA 依赖](https://github.com/vllm-project/vllm/blob/main/requirements/cuda.txt)、[ROCm 依赖](https://github.com/vllm-project/vllm/blob/main/requirements/rocm.txt)、[构建脚本](https://github.com/vllm-project/vllm/blob/main/CMakeLists.txt)。

## 5. 分布式通信：移动激活与专家 token

| 组件 | 传什么／做什么 | 典型场景 |
| --- | --- | --- |
| **NCCL** | GPU 集合通信，例如 AllReduce、AllGather、ReduceScatter | NVIDIA 多卡张量并行、数据并行及分布式推理基础 |
| **DeepEP** | 专家并行的 dispatch／combine，让 token 到对应专家并汇总输出 | MoE Expert Parallel；有高吞吐、低延迟等路径，按版本与网络条件选择 |
| **NVSHMEM** | GPU 发起的通信与共享内存编程能力 | 部分 DeepEP 等通信实现的底层基础；不能将所有版本、所有路径都概括为固定依赖它 |
| **FlashInfer 通信后端** | NVLink 等场景的专家 token 交换，或与专家计算结合 | 当前文档有 one-sided／two-sided NVLink All-to-All，具体支持需核对机器拓扑 |
| vLLM 自定义 AllReduce | vLLM 集成的集合通信优化 | 部分多卡配置下与 NCCL 路径配合使用；不是独立第三方库 |

**DeepEP 与 DeepGEMM 最容易混淆：前者主要解决 token 如何送达专家，后者主要解决专家如何计算。** 典型模块化 MoE 可以组合 DeepEP 与 DeepGEMM／Triton／CUTLASS，但两者并非绑定关系。当前也有把通信和专家计算融合的 Mega MoE 路径，因此上述划分描述主要职责，不是宣称所有实现都严格拆成独立阶段。

来源：[专家并行部署](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)、[MoE 内核与通信兼容表](https://docs.vllm.ai/en/latest/design/moe_kernel_features/)、[PyNcclCommunicator 源码](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/pynccl.py)、[DeepGEMM Mega MoE](https://github.com/deepseek-ai/DeepGEMM#mega-moe)。

## 6. KV Cache 传输、复用与卸载

| 组件 | 主要职责 | 什么时候需要 |
| --- | --- | --- |
| **NIXL** | 推理数据传输抽象，支持 KV Cache 跨实例移动 | Prefill／Decode 分离等，通过 `NixlConnector` 集成 |
| **UCX／GDRCopy** | 通信传输基础与 GPU 内存访问辅助 | 可处在 NIXL 等组件的下层；随传输后端和平台变化，不是所有部署强制需要 |
| **Mooncake Transfer Engine** | KV Cache 的传输与相关缓存基础设施 | 通过 `MooncakeConnector` 等路径集成 |
| **LMCache** | KV Cache 复用、存储、卸载与共享 | 通过 LMCache Connector 接入，可进一步组合 NIXL 等传输组件 |
| FlexKV | 分布式 KV Store 与多层缓存管理 | 官方已有对应 Connector，属于可选缓存系统 |
| MoRI-IO | AMD 平台 KV 数据传输 | 官方有 ROCm 专用 `MoRIIOConnector` |

NIXL、Mooncake、LMCache 不宜视作完全相同层级的互斥替代品。例如 LMCache 的某些部署可以用 NIXL 作为传输基础。

来源：[Disaggregated Prefilling 官方文档](https://docs.vllm.ai/en/latest/features/disagg_prefill/)、[专家并行部署中的 NIXL 安装说明](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/#setup-steps)。

## 7. 量化准备工具：单独登记

| 工具／库 | 作用 | 与推理内核的区别 |
| --- | --- | --- |
| LLM Compressor／compressed-tensors | 模型压缩、量化产物与相关格式支持 | 推理可进一步选择 DeepGEMM、CUTLASS、Marlin 等内核，不应把工具名直接当成 GEMM 后端 |
| NVIDIA Model Optimizer | NVIDIA 模型优化、量化工具链 | vLLM 接收其支持的模型格式，实际计算后端另行选择 |
| TorchAO | PyTorch 量化与优化生态 | vLLM 有相应集成，具体执行随量化类型和设备变化 |
| AMD Quark | AMD 量化工具链 | 与 ROCm 生态相关，当前 ROCm 依赖包含 `amd-quark` |
| AutoAWQ／GPTQModel | AWQ／GPTQ 相关工具链 | AWQ／GPTQ 是算法与格式生态；运行时可能选中 Marlin、Machete 等实现 |

来源：[量化官方文档](https://docs.vllm.ai/en/latest/features/quantization/)、[ROCm 依赖](https://github.com/vllm-project/vllm/blob/main/requirements/rocm.txt)。

## 8. 如何把清单对应到一次实际部署

下面是理解职责的组合示例，不是对任意硬件都成立的默认配置：

- **BF16 dense 模型**：Attention 使用兼容的 FlashAttention／FlashInfer；Linear 使用适用的 PyTorch／基础 GEMM 路径；多卡时再涉及 NCCL 等通信。
- **AWQ／GPTQ 模型**：Attention 仍需单独选择；量化 Linear 可能使用 Marlin／Machete／Humming 等兼容实现。
- **DeepSeek 风格 FP8 MoE**：MLA 使用兼容的 FlashMLA／FlashInfer MLA 等；专家 GEMM 使用 DeepGEMM／Triton／CUTLASS 等；启用专家并行后再选择 DeepEP 或其他通信实现。
- **Prefill／Decode 分离**：在模型计算之外增加 KV Connector，采用 NIXL／Mooncake／LMCache 等适用组件。

确定“我的进程实际用了什么”，应同时记录：vLLM 版本／构建提交、GPU 型号与 CUDA／ROCm、模型与量化格式、KV Cache dtype、TP／EP／DP 配置、启动参数和后端选择日志。若需确认真正执行的 GPU 内核，再看 profiler 调用轨迹；**已安装、可导入、被选中、实际执行，是四个不同层面的证据**。

当前文档提供 `--attention-backend`、`--linear-backend`、`--moe-backend`、`--all2all-backend` 等不同维度的选择接口。参数可用性以所安装版本为准，指定 Attention 后端也不代表同时指定了 Linear、MoE 与通信后端。

来源：[Attention 选择规则](https://docs.vllm.ai/en/latest/design/attention_backends/)、[Linear 后端选择](https://docs.vllm.ai/en/latest/features/quantization/#selecting-linear-backends-per-quantization)、[MoE 后端](https://docs.vllm.ai/en/latest/design/moe_kernel_features/)。
