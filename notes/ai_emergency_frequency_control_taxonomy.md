# 大电网故障后 AI 增强型频率紧急控制：按在线决策路径分类

> 研究范围：严重故障发生后的自适应频率紧急控制。本文按**在线如何由状态产生动作**分类，不按控制资源、神经网络结构或训练算法分类。文献以 IEEE PES Transactions 为主；其他期刊的论文会注明。

## 1. 两条主线

设当前观测状态为 `s_t`，包括可获得的故障前运行方式、故障识别信息和故障后的频率等动态量；紧急控制动作为 `u_t`，可以表示动作量、位置、启动时间或控制器参数。

| 路线 | 在线决策链 | 离线主要学到什么 | 关键区别 |
| --- | --- | --- | --- |
| **A. 状态直接映射到动作** | `s_t → 已训练策略 π → u_t` | 当前状态与合适动作之间的映射 | 在线无需先显式计算未来频率结果，就能输出动作。策略可以由监督学习、模仿学习或强化学习获得。 |
| **B. 状态先映射到结果，再映射到动作** | `s_t → 未来结果或动态关系 → 动作计算 → u_t` | 频率指标、轨迹、风险，或状态与动作对后续响应的影响 | 在线先利用一个可评估未来响应的模型，再决定或调整动作；动作计算可以用规则、另一个学习模型或滚动控制方法。 |

两条线都是**信息处理路径**，不是“深度学习路线”和“强化学习路线”。深度学习可以学习策略，也可以学习预测器；深度强化学习通常用神经网络表示策略。传统固定下垂控制在输入输出形式上也像 A，但若没有学习环节，就不属于 AI 增强型方法。

## 2. 路线 A：学好策略，在线直接输出动作

故障发生后，控制器把可用状态输入已训练策略，输出本次动作。动作既可以是一次性决策，也可以在每次获得新状态后重新推断。因此，“每 0.1 s 更新一次动作”**不会把 A 变成 B**；只要每次仍是 `s_t → π → u_t`，它仍属于 A。

| 论文 | 在线执行的核心步骤 | 分类判断 |
| --- | --- | --- |
| Xie & Sun, 2022, **IEEE Transactions on Power Systems**, *Distributional Deep Reinforcement Learning-Based Emergency Frequency Control* | 离线训练 distributional soft actor-critic（DSAC）；在线由当前状态产生紧急切负荷动作。 | **A**。价值分布用于策略学习，不能据此把它归为“在线显式预测频率轨迹后再决策”。 |
| Chen et al., 2021, **IEEE Transactions on Industrial Informatics**, *Model-Free Emergency Frequency Control Based on Reinforcement Learning* | 对多种紧急场景离线学习策略，在线依据状态给出紧急动作。 | **A**；作为 PES Transactions 之外的直接策略参照。 |
| Brown, Bukhsh & Papadopoulos, 2026, **IEEE Transactions on Power Systems**, *Physics-Shielded Deep Reinforcement Learning for Adaptive Emergency Frequency Control* | 先用神经网络预测节点 RoCoF、识别频率响应相近的负荷组；DDPG 策略给出各组切负荷比例和动作时间，并结合物理屏障。 | **A 为主，带预测辅助的特征构造**。这里的 RoCoF 预测帮助策略组织状态与动作空间，最终动作仍由已训练策略输出；不应仅因存在预测网络就归入 B。 |
| Huang et al., 2022, **IEEE Transactions on Power Systems**, *Learning and Fast Adaptation for Grid Emergency Control via Deep Meta Reinforcement Learning* | 学习能够向新工况和故障情形适应的紧急控制策略。 | **A 的策略适应扩展**；研究对象主要是紧急电压控制，可借鉴方法，不能充作频率紧急控制的直接证据。 |

## 3. 路线 B：评估未来结果，再计算动作

这一主线有两种常见实现。两者都可以随最新测量反复更新。

### B1. 先预测当前趋势，再决定是否以及如何动作

先估计在当前或既定控制下可能发生什么，例如未来频率轨迹、频率最低点或越限风险；再用阈值规则、计算公式或另一个学习模型给出动作。

典型链条：`当前状态 → 预测频率指标 → 与控制目标比较 → 计算动作量`。

Wang et al. (2019) 是一篇**完整的预测—控制方案**：先结合系统频率响应模型与极限学习机（ELM）预测并修正故障后的频率特征，再利用另一个 ELM 及控制目标计算切负荷量。相对地，Lekshmi et al. (2024) 重点是估计故障功率缺额并以神经网络预测频率最低点；若要称为完整的紧急控制器，还须明确预测结果如何触发并决定控制动作。

### B2. 利用学得的动态关系，反复评估与调整动作

离线学习系统动态或动作对后续响应的影响。在线依据最新观测计算本次动作；执行并获得新测量后，再更新控制计算。动作计算过程中可以隐含或显式地评估候选动作的未来效果，**并不要求每次都输出一个可单独报告的“预测最低频率”**。

Cao, Shen & Liu (2024) 属于这一变体：离线辨识 Koopman 动态表示，在线根据测得的状态反复求解反馈控制律、更新功率指令；论文仿真的控制周期为 0.1 s。该论文使用 HVDC 作为执行资源，但在本分类中的位置由其**学得的动态关系如何参与在线动作计算**决定，而不是由资源决定。前一版将它列为第三条并列路线；按“状态→结果/动态关系→动作”的分类主轴，放入 **B2** 更合适。

| 论文 | 学到或估计的中间结果 | 在线如何使用 | 分类判断 |
| --- | --- | --- | --- |
| Wang et al., 2019, **IEEE Transactions on Power Systems**, *Integrating Model-Driven and Data-Driven Methods for Power System Frequency Stability Assessment and Control* | 故障后的频率最低点等动态特征。 | 再结合运行状态和频率目标，由学习模型计算切负荷量。 | **B1：预测与决策串联**。 |
| Lekshmi, Rather & Pal, 2024, **IEEE Transactions on Power Systems**, *Online Estimation of Disturbance Size and Frequency Nadir Prediction in Renewable Energy Integrated Power Systems* | 功率缺额、频率最低点。 | 主要验证在线估计与预测。 | **B1 的预测模块**；不能单独当作已实现动作决策的论文。 |
| Cao, Shen & Liu, 2024, **IEEE Transactions on Power Systems**, *Data-Driven Emergency Frequency Control for Multi-Infeed Hybrid AC-DC System* | 由数据辨识的系统动态关系。 | 在线依据最新状态反复求解控制律、更新指令。 | **B2：学得动态关系＋滚动动作更新**。 |

**前史参照：** Rudez & Mihalic (2016) 在 *IEEE Transactions on Power Delivery* 提出结合宽域测量和短时频率预测的切负荷方法，说明“预测后再动作”早于当前深度学习热潮。其预测控制思想属于 B 的前身，但不能仅凭“有预测”就称其为 AI 方法。

## 4. 不应作为并列分类的维度

1. **一次性动作还是连续更新：** A 可以每个采样周期重新推断；B 也可以每个采样周期重新预测、计算与修正。更新频率是时间组织方式，不是两条主线的边界。
2. **离线训练还是在线学习：** 离线训练、在线推断是 A 与 B 都可采用的部署方式；若部署后还更新模型参数，那是另一项适应机制。“使用新观测重复推断”本身不等于在线学习。
3. **预测器、状态估计器与安全屏障：** 它们可以辅助任一路线。分类时要看最终动作如何生成：Brown 2026 的 RoCoF 预测服务于直接策略；Lekshmi 2024 的最低频率预测尚需连接动作环节。
4. **强化学习还是监督学习、切负荷还是其他资源：** 这些分别是训练方法与执行手段，均不是本分类的主轴。

## 5. 用于后续选题的判断问题

阅读一篇论文时，先追踪**故障发生后每个在线周期实际做了什么**：

1. 已训练模型是否直接输出最终动作？若是，以 **A** 为主；再检查预测、聚类或屏障是否只是辅助模块。
2. 是否先估计未来频率结果或调用学得的系统动态，再由结果决定动作？若是，以 **B** 为主；区分 B1 的指标预测与 B2 的动态关系滚动计算。
3. 论文是否只预测频率最低点、RoCoF 或风险，没有明确的动作生成环节？若是，标记为**控制前端模块**，不要误称为完整紧急控制策略。
4. 对不同工况和未见故障的泛化、测量延迟、动作执行延迟以及动作安全性，论文各自验证到了哪一步？“各种情况都能直接给出有效动作”必须通过专门的分布外测试支持，不能由训练后的快速推断自动推出。

## 参考文献与链接

1. J. Xie and W. Sun, “Distributional Deep Reinforcement Learning-Based Emergency Frequency Control,” *IEEE Transactions on Power Systems*, 37(4), 2720–2730, 2022. DOI: [10.1109/TPWRS.2021.3130413](https://doi.org/10.1109/TPWRS.2021.3130413).
2. C. Chen et al., “Model-Free Emergency Frequency Control Based on Reinforcement Learning,” *IEEE Transactions on Industrial Informatics*, 17(4), 2336–2346, 2021. DOI: [10.1109/TII.2020.3001095](https://doi.org/10.1109/TII.2020.3001095).
3. A. Brown, W. Bukhsh, and P. Papadopoulos, “Physics-Shielded Deep Reinforcement Learning for Adaptive Emergency Frequency Control,” *IEEE Transactions on Power Systems*, early access, 2026. DOI: [10.1109/TPWRS.2026.3706436](https://doi.org/10.1109/TPWRS.2026.3706436).
4. R. Huang et al., “Learning and Fast Adaptation for Grid Emergency Control via Deep Meta Reinforcement Learning,” *IEEE Transactions on Power Systems*, 37(6), 4168–4178, 2022. DOI: [10.1109/TPWRS.2022.3155117](https://doi.org/10.1109/TPWRS.2022.3155117).
5. Q. Wang, F. Li, Y. Tang, and Y. Xu, “Integrating Model-Driven and Data-Driven Methods for Power System Frequency Stability Assessment and Control,” *IEEE Transactions on Power Systems*, 34(6), 4557–4568, 2019. DOI: [10.1109/TPWRS.2019.2919522](https://doi.org/10.1109/TPWRS.2019.2919522).
6. D. Lekshmi, Z. H. Rather, and B. C. Pal, “Online Estimation of Disturbance Size and Frequency Nadir Prediction in Renewable Energy Integrated Power Systems,” *IEEE Transactions on Power Systems*, 39(1), 1126–1137, 2024. DOI: [10.1109/TPWRS.2023.3267744](https://doi.org/10.1109/TPWRS.2023.3267744).
7. Q. Cao, C. Shen, and Y. Liu, “Data-Driven Emergency Frequency Control for Multi-Infeed Hybrid AC-DC System,” *IEEE Transactions on Power Systems*, 39(2), 3880–3894, 2024. DOI: [10.1109/TPWRS.2023.3280074](https://doi.org/10.1109/TPWRS.2023.3280074).
8. U. Rudez and R. Mihalic, “WAMS-Based Underfrequency Load Shedding With Short-Term Frequency Prediction,” *IEEE Transactions on Power Delivery*, 31(4), 1912–1920, 2016. DOI: [10.1109/TPWRD.2015.2503734](https://doi.org/10.1109/TPWRD.2015.2503734).
