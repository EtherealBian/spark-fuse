# 固定输入，改变可以复用的前缀长度 / Fixed-input prefix cut panel

研究 32B 原长输入的收益和数值保真度是否取决于切点。每次都是原有两组提示的相同四条 2,048-token 输入，编辑仅在第 1,536 个 token 之后。比较只复用前 511/512、1,023/1,024、1,535/1,536 token（其余照算），不是为每个切点重新换提示。相邻长度用于观察敏感性，不能据此判定硬件对齐机制或推断未测长度。

单台 DGX Spark，固定 64 层常驻 BF16、SDPA、静态容量 2,080；8 步 greedy。每个切点两组，各 4 个配对，交叉方法先后和分支先后。12 个单元按预定次序，组间逆序访问切点。96 个计时双请求波次（192 个请求）；48 条非计时输出轨迹。计时包括前缀计算、紧凑快照、每次恢复、两次后缀计算及输出回主机。记录快照字节、每对总耗时、准备和分支耗时。诊断不计时：采集全部 BF16 logits；在 CPU 上逐步复算字节、贪心 ID、分差和不同元素数。发生 ID 分岔后，后续 logits 不再视为相同历史的因果比较。

验收分开报告速度、token 和全词表字节。每个切点只有在两组均满足以下条件才记为本面板通过：方法先后两种、分支先后两种分层的配对节省率中位数都至少 3%；每组两条完整 8 步轨迹的全部 BF16 字节相同；所有计时 ID 匹配独立诊断。它不是显著性检验，也不是全局开关。所有从头算的诊断和 1,536 切点还原诊断都须独立命中此前完整轨迹摘要。历史锚点失配使试验不可直接对照，不应被改写为前缀算法失败。不挑选表现好的切点重定义阈值。

English: Each cut reuses fewer or more tokens of *the same four requests*. The fixed two-request end-to-end workload pays snapshot preparation. Neighboring cuts test sensitivity, not a hardware mechanism. All four order-stratified paired medians must improve at least 3%, with independently replayed full-vocabulary BF16 bytes and greedy IDs. Outcomes apply only to this backend, these requests and this eight-token decode budget. No default change. Cold and 1,536-cut anchors must reproduce historical trajectories. This is separate from forced-token layer interventions and fused-restore microbenchmarks.
