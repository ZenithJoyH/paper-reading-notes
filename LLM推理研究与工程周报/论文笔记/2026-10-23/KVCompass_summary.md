# KVCompass：跨区域前缀缓存发现与负载感知路由

## 论文信息

- 完整标题：KVCompass: A Discovery Plane for Internet-Scale Prefix Cache Sharing in LLM Serving。
- 作者：Shiyi Ling、Fei Fang、Chen Qian；UC Santa Cruz。
- arXiv2610.24179v1，提交2026-10-16 19:54:51 UTC／北京时间10-17 03:54:51。正文received10-09；不将收稿日期或抓取日期代替公开版本日期。
- [信息页](https://arxiv.org/abs/2610.24179v1)｜[在线全文](https://arxiv.org/html/2610.24179v1)｜[在线PDF](https://arxiv.org/pdf/2610.24179v1)｜[DOI](https://doi.org/10.1145/3857371)。已在线阅读，未下载PDF/TXT。
- [代码固定快照dd2bf779b66dc850fb8db65d978e3bf54a81184e](https://github.com/011shiyiLing/KVCompass/tree/dd2bf779b66dc850fb8db65d978e3bf54a81184e)，页面10-17；[真机说明](https://github.com/011shiyiLing/KVCompass/blob/dd2bf779b66dc850fb8db65d978e3bf54a81184e/realmachine/README.md)已读。
- [本期周报](../../LLM推理研究与工程周报_2026-10-23.md)。性能区分作者真机、模拟和本文推断，无独立硬件复现。

## 研究背景与动机

前缀缓存由实际服务请求产生，又会随LRU/LFU淘汰、在VRAM和DRAM间移动，持有者不一定由内容哈希预先确定。跨区多副本系统既要找到缓存，又要避免所有热门请求集中到持有者。

只看cache affinity可能增加排队；只找空闲worker可能浪费重算；只看命中长度又可能忽略DRAM reload和WAN延迟。论文把缓存发现与请求转发放在一起优化，局部缓存淘汰仍由worker负责。

## 主要贡献

多级keyless Ludo索引让边缘保有紧凑的全局locator视图；索引记录持有者和存储层级。路由将缓存收益与排队、网络、重新Prefill的代价联合打分。通过真机、事件模拟、机制消融及索引微基准分别验证，不把其中任何一层直接当全部系统证据。

## 方法论

### 发现平面

每个worker报告KV admit、demote/promote、evict事件。maintenance units更新索引，lookup units在edge使用不含完整keys的紧凑视图查询不同长度前缀。层级前缀长度依据Prefill速率和网络代价选择，避免为每个token长度维护同等索引。

### 联合路由

打分考虑剩余Prefill、队列估计、缓存tier reload及WAN距离。负载通过跨区响应piggyback与周期broadcast刷新，EWMA平滑。池接近饱和时提高重算代价，空闲时允许在本地重算以省WAN。

热门前缀可向空闲目标传播，但跨区主要传prompt并重算KV，再放入DRAM；区域内才使用worker间KV拷贝。该方案不证明WAN上直接移动完整KV划算。

### 错误locator不等于错误KV复用

keyless lookup可能对未插入key返回错误locator，且元信息更新可能滞后。选中的worker仍检查自己真实持有的前缀，缺失部分重算。因此错误发现影响路由效率，不应让不同前缀KV被静默复用。跨模型、dtype、parallel partition必须加入兼容性身份，当前主要实验为单模型。

## 实验结果

### 作者公开真机

| 项目 | 条件 |
| --- | --- |
| 硬件／部署 | 三洲14×RTX5090，美8、欧2、亚4；每卡一个worker，三台多卡服务器 |
| 模型／软件 | Llama3.1-8B-Instruct BF16，SGLang0.5.10；完整CUDA/PyTorch/镜像、KV dtype未锁定 |
| WAN | 公网，单向US/EU160ms、US/Asia126ms、EU/Asia149ms；区内loopback TCP |
| 工作负载 | Mixed=50%ToolBench+5%LooGLE+45%APPS；WildChat、LMSYS；4—40req/s，分布图常用16req/s |
| 缺失条件 | 完整ISL/OSL分布、并发和output/total tokens/s定义表；TPOT/ITL绝对表未披露 |

基线为Consistent Hash、Chord、PlanetServe、Central Directory、SkyWalker、Dynamo。仓库中是统一框架下的可切换routing schemes，不将其自动视为各系统完整生产部署。Mixed用LFU，WildChat/LMSYS用LRU，以固定策略隔离路由因素。

最有价值的真机消融为Mixed闭环服务吞吐：移除load awareness下降34%，移除cross-region reuse下降10%，忽略tier下降6%，移除迁移下降4%。这些是各自对完整方案的单项消融，不相加成总收益。

复用token数：hash方案10.6M，KVCompass9.3M，后者服务吞吐更高，说明本实验中负载均衡比单纯最大化复用更重要。论文也报告若干基线P99 TTFT大倍率，但接近饱和程度、基线热点和中心WAN往返都会放大差距，不能当常态统一收益。

附录P/D实测：7P+3D RTX5090，SGLang+Mooncake，24req/s hot-skew Mixed；median TTFT相对各baseline低1.25—1.69×、P99低1.3—15.7×。各方案只改变P路由，D在P同区随机选择，因此不能外推D路由和KV transfer scheduler的贡献。

### 作者索引微基准

双96-core CPU机器；4096-token prompt、五级索引、14-worker打分，forwarding P50/P99=20.6/21.5μs，单线程48k decisions/s。lookup在10M entries视图下P50/P99=310/590ns。这些不包含模型生成和WAN。

十亿entries的32-bit值索引，通过分配和写bucket/classifier arrays测得48bits/entry、6GB；并没有真实插入十亿前缀。部分其他结构是由小规模外推，64-bit值还会更大，不能将其误作十亿完整对象在线服务实测。

### 作者模拟

64 workers/8 regions，模拟A10080GB、Llama3.1-8B BF16，worker设32GB VRAM、128GB DRAM与约25GB/s reload；网络与计算经过校准。论文承认没有相同设置真机对模拟的误差验证。

WAN±100% jitter时P99 TTFT变化小于2%；30%更新丢失时命中率下降2.5个百分点、P99 TTFT上升4.2%；300ms延迟时P99变化约3.4%。这是模拟鲁棒性，不能称公网真机故障注入。实现不自动重传失败更新，没有确定恢复时限。

索引结构误报：95%占用、查询从未插入key时45.3%；真实请求端到端false lookup为Mixed0.10%、WildChat1.27%、LMSYS0.34%。这两组不是同一分母，不能选择较小数字隐去结构误报。

## 局限性

主要为同模型同类worker，异构设备需要各自Prefill/decode代价、缓存格式身份和速率校准。WAN测量与队列估计会过期；模拟误差未量化；跨模型及隔离策略主要在讨论中。跨区数据主权和缓存共享边界需由部署方指定，本笔记不将通用讨论扩展为生态选型。

复现需要先构建native Ludo binding、应用SGLang HiRadixCache tier事件patch，固定chat template/tokenization以保持router/worker前缀key一致。模拟支持Python dict回退，但dict不能作为native Ludo性能复现。数据和模型不随仓库完整打包，需按作者数据准备说明构建；快照README的tag/依赖不足以替代完整镜像锁定。

## 对异构推理测试优化的启发

建议在固定总资源和cache容量下比较RR、cache-only、cache+queue、cache+queue+tier；再给不同设备单独建模Prefill及reload成本。同时记录命中率、实际重算token、排队、P99 TTFT、输出tokens/s和拒绝率，避免命中率升高但热点恶化。

建议4k/1k、16k/1k和客户端并发1/4/64，另做QPS扫描；这些不是论文已有统一实测。可先在单机多worker和人为链路延迟下验证，再做跨机。验证错误locator仅触发重算、更新丢失最终可修复、模型revision及KV dtype不会混用、chat template一致；将真实故障注入与事件模拟分别报告。

## 核心要点

- cache affinity只是路由成本的一部分。
- 单纯更多复用不保证更高吞吐或更低P99。
- keyless发现误报需要worker端真实命中校验兜底。
- 跨区请求路由与跨区搬KV是不同事情。
- 真机、模拟、索引微基准应分别解释和复现。
