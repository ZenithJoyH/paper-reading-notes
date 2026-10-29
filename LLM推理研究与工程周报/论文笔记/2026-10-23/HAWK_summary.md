# HAWK：异构GPU-NPU分离推理的KV传输

## 论文信息

- 完整标题：HAWK: Heterogeneity-Aware KV Cache Transfer for Efficient GPU-NPU Phase-Disaggregated LLM Inference。
- 作者：Junhyeok Park、Jungeun Yoon、Kyungmin Kim、Hunsoon Lee、Shinyoung Ahn、Kangho Kim；主要机构ETRI，第一作者现列Georgia Tech，工作在ETRI完成。
- arXiv2610.29873v1，提交2026-10-21 15:25:35 UTC／北京时间23:25:35；本次阅读2026-10-23。
- [信息页](https://arxiv.org/abs/2610.29873v1)｜[在线全文](https://arxiv.org/html/2610.29873v1)｜[在线PDF](https://arxiv.org/pdf/2610.29873v1)。在线HTML已读，未下载原文附件。
- 作者计划开源，当前论文未提供HAWK专用仓库。Mooncake/TT-Metal是依赖，不是完整实现。
- [本期周报](../../LLM推理研究与工程周报_2026-10-23.md)。以下性能均属作者公开真机结果；没有独立复现。

## 研究背景与动机

P/D分离常按compute-heavy prefill、memory-heavy decode分配资源。换用不同架构的设备，除了执行速度，还会引入通信接口、张量格式与地址映射差异。即使两端都支持RDMA，收下来的字节也未必能被D端attention直接使用。

Tenstorrent NPU将tiled tensor显式分布到多个DRAM channels。传统单连续RDMA buffer可能与这种布局不一致，需要事后重排；还可能把内存压力集中到某些bank，让其他bank有空余时仍发生分配失败。论文要解决的是跨设备数据路径、格式对齐和容量可用性三个问题。

## 主要贡献

一是为NPU增加peer-memory client，使Linux RDMA栈可注册及DMA映射NPU内存。二是分配符合NPU tensor bank分布的多个RDMA buffers。三是将布局变换拆为GPU预变换和转移时映射，兼顾大块传输效率和D端可直接消费的格式。

它不是新的调度器，也没有减少KV本身的逻辑信息量；主要贡献在跨设备搬运实现。

## 方法论

### 直接数据路径

GPU端使用nvidia-peermem，NPU内核驱动实现acquire/get_pages/dma_map/unmap/put_pages等回调。用户态基于Mooncake Transfer Engine，NIC通过共同Linux RDMA栈访问两端device memory，避开host staging。

### 布局感知buffer

NPU端按目标tensor的channel分布分配，每channel一个连续区域，对外表现为一组(pointer,size)。buffer数受channel数约束，额外元信息有限；不能退化为每个小tile单独发RDMA，否则几百字节请求的大量提交会吞掉带宽收益。

GPU将KV从层/位置/head维度重排成以目标channel为最外层的大连续片段，放到GPU RDMA buffer。D把这些片段配对至各channel目标buffer，批量提交操作；完成后重构tensor。P侧仍有变换和buffer拷贝，不能写成全流程零拷贝。可重叠的空间与P侧剩余内存带宽有关，不保证任意prefill负载都能隐藏这些工作。

### 完成与生命周期

D请求KV，P计算/取出KV并发出源buffer元信息，D配对与提交、等待完成、重构，通知P后双方释放相应buffer占用。必须保证设备可见性、请求标识、超时和复用次序正确。论文讨论反向传输和其他算子分离的潜力，但本次核心实验验证的是GPU P→NPU D，不把讨论当完整反向服务实测。

## 实验结果

### 条件

| 维度 | 配置 |
| --- | --- |
| 模型 | Llama3.1-70B，权重148GB；weight/activation/KV dtype未明确逐项列出 |
| P | 4×A100 PCIe40GB，TP4 |
| D | 4×p150a NPU，TP4；同构基线D为4×A100 PCIe80GB |
| 网络 | 两节点，各ConnectX-6，200Gb/s HDR IB |
| 软件 | vLLM0.10.0，Mooncake0.3.6；TT用户态0.68.0、内核2.5.0及作者修改；完整commit/镜像不足 |
| 负载 | ShareGPT Chatbot输入mean/P90=783/899、输出140/148；LongBench Summarization输入2186/2860、输出86/115；Poisson到达率扫描，并发未明确 |
| SLO | Chatbot TTFT/TPOT=2000/200ms，Summarization=15000/200ms；报告90%达标点 |

对照为GG-Host、GN-Host、GG-Direct、GN-HAWK。另有“GN direct但连续NPU buffer”原型因内存bank不均衡／碎片化OOM而退出量化比较，不能从其缺失推断所有收益都来自其中一项改动。

### 真机性能

- 90%联合SLO达标时，GN-HAWK相对GG-Direct的输出吞吐低0.48%（Chatbot）、7.88%（Summarization）。Chatbot两者可到0.65req/s；Summarization异构0.60、同构0.65req/s。
- P90 TTFT单项SLO对应请求率差为2.90%/5.90%，不是上项联合SLO的同一数字。
- 大传输下接近GPU direct传输延迟；微基准覆盖10—1280MB，计时从请求KV到写入cache pool，不能当TTFT或TPOT。
- 替换同构D为H100的Table9，输出tokens/s分别95.96/54.28，对照HAWK89.69/50.75；这组不能用来冒充A100比较的绝对基线。
- 逐点TTFT/TPOT绝对表、ITL、P99、total tokens/s未完整披露；不估造读图值。

### 能效／成本必须注明估算成分

Table8的ASIC能效：Chatbot GG67.09、GN84.48 output tokens/s/kW；Summarization41.01、47.18，对应1.26×/1.15×。NPU监控只给core功率；GPU用DCGM板卡功率结合NVBit和功率模拟器估计ASIC份额。这是实测服务吞吐与估计功率的混合，不是同口径整机能耗测量。

成本效率为1.19→2.05、0.72→1.16 output tokens/s/$1000，最高1.72×；按NPU官方价及GPU市场价估计，不含完整服务器/运营成本，不构成当前采购建议。

## 局限性

单模型、TP4、小规模两机；NPU模型栈绑定设备及卡数。需要驱动改动且作者实现未开源，复现门槛高。未覆盖国产NPU、长输入16k以上、生产级并发/故障恢复；也没有保证移植到另一个blocked layout仍得到相同收益。P侧变换、buffer复制及请求控制流程仍有代价。

## 对异构推理测试优化的启发

建议先对本机tensor的逻辑shape、padding、物理bank与KV page映射建立可核验说明，再测host staging→direct+post-convert→layout-aware transfer。微基准与服务实验分开，记录提交数量、变换耗时、实际传输、D重构和排队。

4k/1k、16k/1k，客户端并发1/4/64及独立QPS扫描是建议，不是作者已有实测。正确性以逐层KV/首步logits及任务输出验证，稳定性覆盖块复用、取消重试、超时回收、bank碎片化和长时运行。直接设备传输接口不开放时，先优化host staging路径，不假设方法可直接部署。

## 核心要点

- KV布局、设备内存分配与RDMA请求组织需要协同。
- 设备间直连并不自动消除布局转换与device内拷贝。
- 90%联合SLO、P90 TTFT单项和传输微基准是不同指标。
- 论文的功率/成本效率含估算，不能称整机独立实测。
- 最直接的下步证据是实现开源、其他设备layout适配及故障回归。
