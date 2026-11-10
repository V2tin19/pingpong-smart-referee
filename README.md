# 乒乓球视觉智能裁判系统 (Smart Table Tennis Referee)

602Lab 乒乓球视觉智能裁判系统的**共用裁判核心层（Core Engine）与硬件对接仓库**。  
依据组长下发的权威方案《TTNet、TrackNet、YOLO 共用计分状态机方案》实现。系统负责接收前端多源感知算法（YOLO / TrackNet / TTNet）产出的**统一逐帧观测（Observation）**，并结合**物理硬件实体按键**，完成全自动轨迹滤波、物理事件判定、球台落点单应性解算、6状态计分流转与转播级比分看板渲染。

> 2026-11-10 接手审查与实测结论见 [健康检查与部署优化报告](docs/project_health_20261110.md)。
> 当前保留默认 `imgsz=960`；两个机位完整标注实测表明，640/512 明显损失召回，832 也未满足跨场景零退化。
> 运行入口目前支持本地视频文件/JSONL；相机、RTSP 和 GPIO/UART 驱动仍是下一阶段工程，不是已交付能力。

---

## 一、 系统架构

```text
┌─────────────────┐       统一逐帧观测 (§7.1)
│ 前端视觉感知模组 │ ─────────────────────────────────┐
│(YOLO/TrackNet等)│  JSONL 观测流 (坐标/置信度/可见性)  │
└─────────────────┘                                  ▼
                                         ┌───────────────────────────┐
                                         │   轨迹平滑与物理事件提取   │
                                         │ (时钟驱动卡尔曼/反折/拐点) │
                                         └─────────────┬─────────────┘
┌─────────────────┐  GPIO 中断 / 串口                  │
│ 裁判物理实体按键 │ ───────────────────┐              │ 物理事件 (§7.4)
│ (纠错/重发/加分) │                    │              ▼
└─────────────────┘                    ▼   ┌───────────────────────────┐
                               ┌───────────┤   6 状态乒乓球计分状态机   │
                               │           │  (ITTF 5 条核心判罚规则)  │
                               │           └─────────────┬─────────────┘
                               ▼                         ▼
                    ┌─────────────────────┐   ┌────────────────────────┐
                    │ 物理硬件 LED 数显屏 │   │ 转播 HUD 画面渲染看板  │
                    │ (串口/UART 比分外发)│   │ (比分牌/轨迹/2D俯视雷达)│
                    └─────────────────────┘   └────────────────────────┘
```

### 核心设计原则
1. **职责严格解耦**：适配器只回答“模型看到了什么”，处理器只回答“发生了什么物理事件”，状态机只回答“谁该加分”，各层互不越权。
2. **核心层零深度学习依赖**：`core/` 核心引擎采用标准库 + NumPy 实现，**完全不依赖 PyTorch / CUDA**。在嵌入式单板或低功耗工控机上，纯 CPU 运行速度高达 **15,000+ FPS**。
3. **数据契约自校验**：严格实现方案 §7.1 与 §7.3 红线（漏检不得填位、坐标还原到原图、时间戳归属被预测帧），非法状态在构造时立即抛错，杜绝隐性污染。
4. **四条参考门控，宁漏勿误**：发球准入（本方→对方落台序列，颠球=准备动作）、发球前击球校验（无球拍接触挡下）、回合内击球要求（同侧两弹夹击球=序列误拼不计分）、判分后 2s 冷却与静默默认不判分。每道门的触发计数随 `describe()["gates"]` 暴露，可量化、可消融（详见 `docs/state_machine_rules.md` 与 `docs/decisions.md` R-022）。
5. **瞬移零容忍（感知降噪 P1）**：关联速度硬截断（隐含速度超物理上界即断轨，速度制、帧率无关）+ 确认帧后候选选择距离门控 + 可选球台地板掩码（`--reject-floor`）——检测瞬移在进入卡尔曼之前被拦下，杜绝虚假击球（R-024/R-025）。

---

## 二、 目录结构

```text
pingpong/
├── core/                         # 裁判核心大脑（零外部深度学习依赖，纯 CPU 秒级运行）
│   ├── schemas.py                # 统一数据契约 (FrameObservation / RefereeEvent)
│   ├── state_machine.py          # 6 状态计分状态机 (5 条规则 + 四条参考门控, 全部单独开关)
│   ├── trajectory_processor.py   # 轨迹处理与事件提取 (时钟卡尔曼滤波/击球/落台)
│   ├── geometry.py               # 球台几何与四套机位预设 (test2 / game2 / rally / plos)
│   ├── homography.py             # 球台单应性变换 (像素坐标 ──> 真实球台 2D 厘米落点)
│   └── base_adapter.py           # 适配器抽象基类
├── adapters/                     # 各模型输入适配器
│   └── yolo_adapter.py           # YOLO11n 感知适配器
├── visualizer/                   # 转播级比分板与可视化
│   └── broadcast_hud.py          # 左上角比分牌 + 轨迹尾迹 + 2D 俯视落点雷达
├── weights/                      # 预训练微调小目标权重
│   └── yolo11n_clean_best.pt     # 乒乓球专属轻量权重 (5.4MB)
├── examples/                     # 免视频、免模型的即用型观测测试数据
│   ├── sample_tracknet_600f.jsonl# TrackNet 分支 600 帧真实比赛观测流
│   └── sample_yolo_300f.jsonl    # YOLO 分支 300 帧真实比赛观测流
├── tests/                        # 自动化测试套件 (执行命令见下文，以当前结果为准)
│   ├── test_contracts.py         # 数据契约校验测试
│   ├── test_state_machine.py     # 状态机判罚规则与四条门控单测 (含生活流长会话回归)
│   ├── test_trajectory_processor.py # 轨迹平滑、事件提取与瞬移守卫单测
│   ├── test_homography.py        # 物理落点解算单测
│   └── test_yolo_adapter.py      # 适配器纯逻辑单测 (地板掩码, 免 torch)
├── tools/                        # 测量与分析工具 (需视觉环境)
│   ├── compare_weights_benchmark.py   # 权重精度/误报基准对比 (对照官方真值)
│   ├── analyze_gt_envelope.py         # 真值包络与速度分布 (掩码/截断安全边界)
│   └── analyze_fp_locations.py        # 误检位置/置信度/bbox 尺寸分析 (降噪依据)
├── docs/                         # 项目技术文档规范
│   ├── hardware_interface.md     # 【重点】硬件与外设对接规范 (相机/按键/串口/投屏)
│   ├── data_contract.md          # 统一数据契约 (§7.1 与 §7.4 详细定义)
│   ├── state_machine_rules.md    # 6 状态机流转图与判罚规则细则
│   ├── gates_validation.md       # 四条参考门控的实现与验证记录 (Test4 复放对照)
│   ├── proposal_review.md        # 《状态机修改方案》(参考文档) 评审意见：采纳/暂缓/不采纳及理由
│   ├── decisions.md              # 25 条工程实现决策与风险记录 (R-001 ~ R-025)
│   └── upstream/                 # 组长下发原始方案 (只读参考)
├── run_referee.py                # 系统统一运行入口
├── requirements.txt              # 基础运行环境 (仅 numpy + opencv)
├── requirements-vision.txt       # 视觉推理选配依赖 (ultralytics + torch)
└── .gitignore                    # 严密过滤大文件与临时视频
```

---

## 三、 快速上手

### 1. 环境安装
```bash
# 基础核心环境（状态机、事件提取、转播显示）
pip install -r requirements.txt

# [可选] 如需本地运行 YOLO 视频目标检测，安装视觉依赖
pip install -r requirements-vision.txt
```

### 2. 模式 A：观测流极速评测（无模型、纯 CPU，0.05秒出分）
适用于消费 TrackNet / TTNet / 外部感知模块导出的 `.jsonl` 观测文件：

```bash
# 运行内置的 TrackNet 600 帧比赛观测流
python run_referee.py --jsonl examples/sample_tracknet_600f.jsonl --preset test2

# 运行内置的 YOLO 300 帧比赛观测流
python run_referee.py --jsonl examples/sample_yolo_300f.jsonl --preset game2
```

**终端输出示例：**
```text
========================================================================
【乒乓球智能裁判】观测流评测模式 (JSONL Stream)
观测文件: sample_tracknet_600f.jsonl
机位预设: test2 (球台ROI net_x=929)
========================================================================
  [帧 0070 |  0.583s] 事件: RACKET_HIT  半台: A       ID: racket_hit_00001
  [帧 0111 |  0.925s] 事件: BOUNCE      半台: B       ID: bounce_00001
  [帧 0147 |  1.225s] 事件: RACKET_HIT  半台: B       ID: racket_hit_00002
  [帧 0179 |  1.492s] 事件: BOUNCE      半台: A       ID: bounce_00002
  ...
========================================================================
处理完成！总帧数: 600 帧 | 耗时: 0.039s (15504.7 FPS)
物理事件统计: {'RACKET_HIT': 9, 'BOUNCE': 6}
最终比分: P1 (半台A) 0 : 0 P2 (半台B)
========================================================================
```

> 该样例从回合中段开始（无发球序列），门控 1（发球准入）正确地**不给分**：
> 早期版本曾在这里误判 `SERVE_FAULT` 送出 0:1。证据不足时默认不判分——
> **宁可漏判，不可误判**。

### 3. 模式 B：端到端视频推理并渲染比分看板
适用于本地视频文件，调用 YOLO 适配器推理并渲染转播视频（相机输入尚未接入）：

```bash
python run_referee.py \
    --video data/test_2.mp4 \
    --weights weights/yolo11n_clean_best.pt \
    --preset test2 \
    --output output/demo_referee.mp4
```

### 4. 输入尺寸与硬件优化评测

使用已有视觉环境执行（只做检测评测，不渲染/降采集帧率）：

```bash
python tools/benchmark_input_sizes.py --video data/test_4_50s.mp4 --annotations data/test_4.zip --weights weights/yolo11n_game1_best.pt --preset test4 --sizes 960 832 768 640 512 --device 0 --output output/input_sizes.json
```

- 不传 `--sample-count` 默认评测全部可用标注；抽样只用于冒烟，不用于验收。
- 缺失标注不当作无球；负坐标仅为不可见/不可定位标注，不代表已人工确认无球。
- 输出原图像素误差、Recall@5/10/25px、预热后适配器耗时及逐帧结果。
- `--half` 是独立 FP16 实验；生产入口也支持 `--device` / `--half`，但默认仍为 FP32、960。
- 单机稀疏帧耗时不等于实际部署端到端 FPS，必须补目标硬件连续录像与判罚回放验收。

### 5. 运行完整自动化测试
```bash
python -m unittest discover -s tests -p "test_*.py"
# 预期输出：Ran ... tests ... OK（数量随新增回归更新）
```

---

## 四、 硬件与外设对接指南

如果你是负责**硬件搭建、按键盒、串口数显屏或工控机部署**的同学，请直接参阅：
👉 **[硬件与外设对接规范文档 (docs/hardware_interface.md)](docs/hardware_interface.md)**

文档内涵盖：
* 工业相机采集指标（分辨率、曝光时间、安装倾角）；
* 实体裁判按键 GPIO/串口接入与重发球（LET）、误判纠错回调；
* 外接数码管 / 电子记分牌 UART 串口帧协议；
* 嵌入式边缘盒子（RK3588 / Jetson Orin / 树莓派）低延迟部署考量。
