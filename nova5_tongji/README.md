# Nova5 × 同济 SuperPower：传送带标记跟踪

当前发布版本：[`v0.1.1`](RELEASE_NOTES_v0.1.1.md)，基于首次实时追踪实现完成
关节抖动修复，并已获得现场“不再抖动”确认。

独立工程，复用 [同济 sp_vision_25](https://github.com/TongjiSuperPower/sp_vision_25)
原版 C++ `tools::ExtendedKalmanFilter`，通过外层适配连接二维码/ArUco、Nova5
正运动学、手眼标定、PBVS 和 ServoJ。仓库原有仿真与 `nova5_visual_servo` 文件不变。

**当前状态：原版 EKF 和标准自瞄 `Aimer` 已在 Ubuntu 编译并通过单元测试；
同济 Aimer/Nova5 六轴实时 `ServoJ` 追踪已在实体 Nova5 + D435i 上运行，最新版本
经现场确认不再出现明显关节抖动。200 ms 回正仍未作为稳定指标验收。**

2026-11-17 Orin 验证：13 项中 12 项通过（含 ArUco）；普通 QR 解码测试因系统 OpenCV
未链接 QUIRC 而失败。需要带 QR 解码能力的 OpenCV 才能使用 `kind=qr`。
现场原控制程序使用直接 V4L2/TCP，目前与本工程的 ROS2 执行入口尚未接通。

## 原版保留范围

上游作为 Git 子模块完整保留，固定提交：
`bd9f5e798fa3c6dd3b483ae6627796afb41c608d`。本工程不修改其源码。
构建直接编译上游 `tools/extended_kalman_filter.cpp`，不是另写一份 Python KF。

| 部分 | 处理 |
| --- | --- |
| EKF 状态传播、增益和 Joseph 协方差更新 | 直接使用同济原版实现 |
| 原装甲板识别、整车旋转模型、TinyMPC 规划器 | 保存在子模块；二维码执行路径不调用 |
| 单标记观测/运动模型 | 新增 `src/conveyor_adapter.cpp` |
| 相机、正运动学、手眼变换 | 外层适配，使用时间配对关节状态 |
| 原版 `auto_aim::Aimer` yaw/pitch 指令 | 由 `TongjiAimer` 直接调用编译后的上游实现 |
| yaw/pitch 到 Nova5 六轴 TCP | 新增 `tongji_controller.py`，保持 40 cm 距离并经 Nova5 雅可比发送 |

原版 `Target` 使用旋转中心、半径、多块装甲板等状态，`update()` 中还有取最近三块板
的循环，因此不能简单设置“装甲板数=1”来处理二维码。原云台 yaw/pitch 规划器也不能直接
输出六轴关节位置。本工程保留原版供对照，明确增加必要适配，不声称整条自瞄链可原样运行。

“同济控制器”指上游 `tasks/auto_aim/aimer.cpp` 的命令生成路径。上游 TinyMPC
`Planner` 是离线可行轨迹求解器，官方说明其输入不包含云台实际状态，因此本二维码入口
不把它标成闭环控制器。当前单标记入口实际调用原版 `Aimer`，关闭弹道与开火语义，
再把 yaw/pitch 光轴转换成 Nova5 相机位姿。投影弹道使用很大的等效弹速，使重力补偿
在 0.4 m 处可忽略；这是明确的光轴伺服适配，不是发射控制。

## 模型

状态为 `[p_B(3),v_B(3),yaw_B,yaw_rate_B]`，速度未知时在线估计。
`T_AB` 表示把 B 坐标转换到 A 坐标，单位 m / rad / s。

传送带部分按用户提供的匀速模型：

```text
x_k = x_(k-1) + v_(k-1) * Δt
v_k = v_(k-1)
```

这里 x 是传送带方向上的位置；传送带方向尚未标定时，在 Base 的三个平移轴分别使用
同一个模型，即 `p_k=p_(k-1)+v_(k-1)Δt`。30 FPS 的名义 Δt 为 1/30 s ≈ 33.333 ms，
实际计算使用相邻图像时间戳之差；不把掉帧后的 66.7 ms 仍当成 33.3 ms。
例如 v=0.1 m/s 时，一帧预测前进约 3.33 mm。速度由观测估计，不填入虚构的已知带速。
过程噪声只用于容纳微小速度变化/模型误差，不等于增加恒加速度状态模型。
图像像素通过未来相机位姿和投影计算，不对像素直接套固定 vΔt。

```text
标记四角 → PnP → T_CQ
关节时间配对 → FK(q) × T_EC → T_BC
同济 EKF + 单标记非线性观测 → 目标状态/速度
未来目标位姿 + 未来相机位姿 → 图像预测（0 / 50 / 100 ms）
目标相对位姿（50 cm） → 既有六轴 PBVS → 限幅 → ServoJ
```

观测函数为 `[X_C/Z_C,Y_C/Z_C,Z_C,yaw_B]`。它有四个残差，满足原版 EKF
诊断代码访问四项残差的约定。滚转/俯仰保持最近接受观测，**不提供完整三轴角速度预测**。
前馈包含线速度和基坐标系 yaw 角速度。

原版 EKF 内的诊断 NIS 是更新后的残差统计，本适配另算**更新前 NIS**作异常拒绝，
不把原版诊断阈值直接当成门控。PnP 的位置、深度、姿态相关，观测噪声仍需实测标定。
重复预测使用滤波器副本，不累计外推；拒绝测量不刷新有效观测时间。

图像预测时域相对于图像时间戳，未来相机采用受限 `q+qdot H` 模型；这不是 ServoJ
动态模型，也不把 255 ms 等效跟随滞后直接当作预测时域。ROS 图像与关节时间必须同钟，
反馈采样时间是否真实由配置明确记录，未核验时不能开始运动。

## Ubuntu 22.04 / Orin 构建

```bash
git submodule update --init --recursive
sudo apt install build-essential cmake libeigen3-dev python3-numpy python3-opencv
bash nova5_tongji/scripts/build.sh
bash nova5_tongji/scripts/test.sh
```

Python OpenCV 必须带 `aruco` 模块；若系统包不带，在独立环境安装 `requirements.txt`。
不需要编译整个上游的 OpenVINO/工业相机/串口依赖，因为此次实际复用的是原版 EKF 核心。
完整上游原应用仍按其 README 单独构建。

Windows 可用 MinGW 和 Eigen 源码：

```powershell
cmake -S nova5_tongji -B nova5_tongji/build -G "MinGW Makefiles" -DEIGEN_INCLUDE_DIR=C:/path/to/eigen
cmake --build nova5_tongji/build -j 2
$env:PYTHONPATH="$PWD/nova5_tongji/python;$PWD/nova5_visual_servo"
python -m unittest discover -s nova5_tongji/tests -v
```

## 相机预览和预测

需 ROS2 Humble、Dobot 驱动、相机图像/CameraInfo、`cv_bridge`、SciPy，
以及已构建/安装的本仓库 `nova5_visual_servo` 包（提供 URDF 和原 ServoJ 节点资源）。
在已有 colcon 工作区安装该包并 source `install/setup.bash`；本新工程用脚本直接启动。

```bash
cp nova5_tongji/site.example.json nova5_tongji/site.local.json
# 填入现场 T_EC、关节和工作空间限制、相机 optical_frame、时间戳核验结果。
# 示例标定是占位值，site_verified=false，预览只显示检测。
bash nova5_tongji/scripts/run_preview.sh --ros-args \
  -p config:="$PWD/nova5_tongji/site.local.json" \
  -p kind:=aruco -p marker_id:=0 -p side_m:=0.1 \
  -p preview_host:=0.0.0.0 -p preview_port:=8765
```

真正 QR 码使用 `-p kind:=qr -p target_text:=nova5-target`。
默认 ArUco `DICT_4X4_50`、ID 0、边长 100 mm，与此前日志中标记类型一致；实际边长必须核对。

网页 `http://<Orin当前IP>:8765/`：绿框为检测，青色为当前估计，黄色/紫色为
50/100 ms 预测。预测须有标定和时间配对关节反馈。网页与检测共用相机图像，JPEG 异步编码。
订阅 `/tongji/prediction` 查看状态、速度和预测像素。

默认话题：

- 图像 `/camera/camera/color/image_raw`
- 内参 `/camera/camera/color/camera_info`
- 关节 `/joint_states_robot`
- 标记观测 `/visual_servo/target`

可通过 ROS `-r 原话题:=现场话题` 重映射。ROS 入口使用本仓库现有 ROS2/Dobot 驱动接口。
已有 V4L2/TCP 现场运行环境可使用下述独立入口。

## 已有 V4L2/TCP 运行环境

`tcp_bridge.py` 把原版同济 C++ EKF 接入已核对的现场执行链，保留其 PBVS、
制动限制器、ServoJ、手眼标定、旋转滤波和停止机制。当前仅替换平移估计与线速度来源；
尚未实现新的漏检接管或角速度前馈。它要求指定版本的 `realtime_tracker.py` SHA256，
版本不同直接拒绝启动，不能用于任意旧工程。

```bash
export TONGJI_LEGACY_PROJECT=/absolute/path/to/verified/visual-servo
export TONGJI_TCP_MODE=shadow
bash nova5_tongji/scripts/run_tcp.sh --config /absolute/path/to/verified.yaml \
  --experiment-dir /absolute/path/to/new-experiment --motion-limiter braking --no-display
```

Shadow 不允许 `--execute` 或 `--auto-arm`。受控实机试验才使用
`TONGJI_TCP_MODE=control` 和这两个参数；同一时间只能有一个相机读取者和机械臂指令源。
初始化或不可信估计产生零反馈参考；新旧位置预测相差超过 30 mm 会停止。
`tongji.jsonl` 单独记录新估计，旧执行日志的目标估计字段仍来自原估计器，不能混用。
时间戳仍是图像到达代理，不能宣称完成真实采集时间标定。

全局速度倍率的独立对照可显式设置 `TONGJI_GLOBAL_SPEED_PERCENT=100`。
旧执行环境启动时会把 `SpeedFactor`、`VelJ`、`AccJ` 一起写成配置值；此覆盖器在启动后、
运动前只重设 `SpeedFactor`，通过新反馈核验全局倍率与未改变的另外两项比例。
命令拒绝、非空闲状态或反馈核验失败即终止。`speed_scaling.jsonl` 保存反馈记录。
覆盖器默认不启用，也不会更改已保存的旧配置或默认 P 增益。
2026-11-17 的三组单轮对照未显示稳定提速，仍未达到 200 ms 回正；结果见
`validation/global_scaling_20261117.json`。该结论不能解释为所有运动模式均不受倍率影响。

`TONGJI_RESPONSE_PROBE=1` 是明确选择的固定靶标诊断模式：6 秒等待、1° 相对姿态偏置
保持 8 秒、恢复原姿态，26 秒结束。保持 50 cm 配置距离，不直接发送关节阶跃。
额外限制 TCP 位移 30 mm、单关节累计位移 0.08 rad、参考关节速度范数 0.15 rad/s。
触发限制即沿原执行链停止；日志为 `response_probe.jsonl`。靶标或传送带在期间移动时，
该轮不能用于固定目标增益比较。这些边界属于诊断实验，不代表机器人额定安全限制。
诊断结束会退出控制程序，其内置视频服务随之结束；持续看视频应启动独立只读预览。

### 实际像素偏差试验

`TONGJI_PIXEL_PROBE=30` 与姿态探针互斥，默认关闭。它先在原增益 0.5 下居中，
再改变相机相对目标的平移参考，使真实标记在画面里偏移；到达准备窗口后恢复居中参考，
以配置中的 P 增益回中。不是修改图像或伪造像素误差。
`pixel_probe.jsonl` 记录实际起始误差、指令切换时间和图像误差。

误差相对于标定主点 `(cx, cy)`，不等同于图像几何中心；时钟仍使用主机接收代理。
默认验收为误差范数连续 300 ms 不超过 5 px，报告首次进入该持续区间的时刻。
准备阶段容许 X 偏差 ±7 px，较大幅度允许 Y 偏差 10 px，因此必须报告实际起始误差。
初始化、偏移和回中均有超时，回中超时不是成功。

10–60 px 使用 40 mm / 0.12 rad 位移边界。大于 60 px 时先用现场运动学、当前反馈与
手眼标定求解整条居中及偏移路径，检查关节余量、雅可比和连杆原点高度；再限制在
该路径的关节范围及 20 mm 横向走廊内。这不是场景碰撞检测，必须由操作者确认空间已清空。
大幅度探针将关节参考速度范数限制在 0.20 rad/s，单独记录限幅前需求；原制动限制器与
视觉失效停止规则继续有效。代码上限 400 px，不代表该范围已经实机验收。

2026-11-17 现场探索：30/60/100/150/220 px 请求已进行回中测试；300 px 请求达到
约 295 px 后，回中横向轨迹偏离超过 20 mm，保护停止，未继续增大，也未驶出画幅。
小偏差下 P=2.0 的单轮结果约 1.06 s 进入 5 px 稳定带，200 ms 目标未达到。
P=2.0 只经过小偏差探针验证，未提升为生产默认值。不同幅度是单轮探索，不是统计验收。
详情见 `validation/pixel_response_20261117.json`。

```bash
python3 nova5_tongji/scripts/analyze_pixel_probe.py /absolute/path/to/experiment
```

### 参数梯次扫描

`run_gradient_sweep.py` 面向当前已验证的 V4L2/TCP ArUco 执行链，自动逐组
覆盖 `gain`、`angular_gain` 和 `speed_factor`。它要求 Dobot 驱动、D435i 相机和
单标记 tracker 已运行，并会在每轮自行启动、arm/start、采样和停止执行器。

默认只打印组合；显式加 `--execute` 才会运动：

```bash
python3 nova5_tongji/scripts/run_gradient_sweep.py \
  --gains 1:4:1 --angular-gains 0.5:1:0.5 --speed-factors 30 \
  --duration 8 --execute
```

`1:4:1` 表示包含端点地生成 `1, 2, 3, 4`，多个点也可写成 `1,2,4`。脚本默认每轮
暂停，要求把标记放回相同初始偏移并确认工作空间；非交互运行可加
`--no-prompt-between-runs`。结果目录包含：

- `summary.csv` / `summary.json`：各组合的命令数、误差均值/P95、最终误差、
  ServoJ 往返、发送周期、最大关节速度和累计位移
- `trial_NNN/tracking.jsonl`：执行器原始日志
- `trial_NNN/command.json`：本轮完整命令和参数

每轮默认限制线性速度 5 cm/s、累计平移 50 mm、累计旋转 5°；脚本不会自动提高这些边界。

当前 ArUco 执行器支持误差—速度自适应增益：默认误差从 12 px 到 30 px 时逐步提高，
目标图像速度从 60 px/s 降到 20 px/s 时也会逐步提高相同权重。大误差且慢目标时，
增益从基础值升至 `large-error-gain`，线性速度上限从基础值升至
`large-error-max-linear-speed`；快速目标仍坚持基础增益。

```bash
python3 nova5_tongji/scripts/run_gradient_sweep.py \
  --adaptive-gain --gains 2.4 --large-error-gains 4:8:2 \
  --angular-gains 1 --speed-factors 30 --execute
```

也可在直接启动执行器时使用 `--adaptive-gain`，并通过
`--large-error-start-px`、`--large-error-full-px`、
`--slow-target-full-px-s`、`--slow-target-zero-px-s` 调整过渡区。

现场移动目标优先使用
`/home/clover/ur_ws/ibvs_rm65/config/nova5_fast_tracking.yaml`。该配置开启目标图像
速度前馈，平移基础限速为 10 cm/s，大误差上限为 16 cm/s，ServoJ 时间片为 20 ms；
单标记旋转只使用 0.5 增益和 0.15° 单步上限，避免角度噪声耗尽运动预算。

同一配置还提供静态回中冲刺：目标图像速度低于 8 px/s 且误差达到 4 px 时，将
`ServoP` 单步时间片临时提高到 100 ms，并使用最高 15 cm/s、单步不超过 20 mm 的
平滑笛卡尔修正。它用于一次消掉大部分静态居中误差，而不是反复执行 20 ms 小步。

```bash
ros2 run ibvs_ur5 nova5_low_speed_executor.py \
  --config /home/clover/ur_ws/ibvs_rm65/config/nova5_fast_tracking.yaml \
  --execute --output-log /tmp/nova5-fast-tracking.jsonl
```

启动后仍需显式调用 `/nova5_ibvs/arm_execution` 和
`/nova5_ibvs/start_execution`。前馈只补偿观测到的目标速度，不等于替代位置闭环。

## 机械臂执行入口

### Nova5 实时追踪控制器（12D 全姿态 ESKF / 8D Aimer）

直连入口默认使用 12 维完整位姿误差状态 EKF，用预测的 PnP 位置、姿态和角速度生成
定距正对相机位姿；传入 `--ekf 8d` 可回退到原同济 EKF + Aimer 中心追踪路径。
12D 模式要求 `align_marker_plane: true`。可用 `--upright` 要求二维码解码 X 轴对齐
图像右侧，或用 `--no-upright` 只保持正对平面。
可先运行 `bash nova5_tongji/scripts/run_tongji_controller.sh --ekf 12d` 检查所选模式。
直连入口按消息到达时的关节反馈转换 PnP 观测，手眼运动时的时间配对精度仍需实测。

该入口位于本目录；12D 模式直接使用完整姿态 ESKF，8D 兼容模式复用原版同济
EKF 和 `Aimer`。官方 Dobot Python SDK 只作为 50 Hz 传输层。启动前必须关闭
`dobot_bringup_v4`，否则 29999/30004 端口会冲突。
D435i 相机和 `real_tag_tracker.py` 应已运行，并在 `/tag_tracker/pose_camera`
发布 ID 0、边长 100 mm 的 ArUco 位姿。

#### 完整启动流程

以下命令按三个终端执行。终端中的命令可以整段复制。三个终端都先执行基础环境：

```bash
cd /home/clover/Desktop/ur5-pbvs-mujoco-simulation
source /opt/ros/humble/setup.bash
source /home/clover/ur_ws/install/setup.bash
```

终端 1，启动 D435i 彩色相机。当前实时追踪使用 `424x240x30` 彩色流：

```bash
cd /home/clover/Desktop/ur5-pbvs-mujoco-simulation
source /opt/ros/humble/setup.bash
source /home/clover/ur_ws/install/setup.bash

ros2 launch realsense2_camera rs_launch.py \
  camera_namespace:=camera \
  camera_name:=custom \
  device_type:=d435i \
  rgb_camera.color_profile:=424x240x30 \
  rgb_camera.enable_auto_exposure:=false \
  enable_depth:=false \
  enable_sync:=false \
  align_depth.enable:=false \
  enable_gyro:=false \
  enable_accel:=false
```

终端 2，启动 100 mm、`DICT_4X4_50`、ID 0 的 ArUco tracker：

```bash
cd /home/clover/Desktop/ur5-pbvs-mujoco-simulation
source /opt/ros/humble/setup.bash
source /home/clover/ur_ws/install/setup.bash

NOVA5_UPRIGHT_FAST_DETECT=1 ros2 run ibvs_ur5 real_tag_tracker.py \
  --board-type marker \
  --dictionary DICT_4X4_50 \
  --tag-id 0 \
  --tag-size 0.1 \
  --base-frame '' \
  --color-topic /camera/custom/color/image_raw \
  --camera-info-topic /camera/custom/color/camera_info \
  --no-enable-depth \
  --no-publish-debug-image \
  --async-processing \
  --no-show \
  --print-every 1000 \
  --frame-log /home/clover/ur_ws/ibvs_rm65/runtime/nova5_tracking_frames_single.jsonl
```

`NOVA5_UPRIGHT_FAST_DETECT=1` 将 adaptive-threshold 扫描步长从 2 提高到 10，
用于降低 424x240 图像的检测耗时。当前空场景 detector 路径约 3 ms；二维码在画面内
时的端到端帧率仍需现场复核。如果现场光照变化导致 `marker_not_found` 增加，
应回退到精细扫描。

终端 3，先检查视觉话题，再启动实时追踪：

```bash
cd /home/clover/Desktop/ur5-pbvs-mujoco-simulation
source /opt/ros/humble/setup.bash
source /home/clover/ur_ws/install/setup.bash

ros2 topic echo --once /tag_tracker/pose_camera

observation_log="/home/clover/ur_ws/ibvs_rm65/runtime/tongji_12d_observations_$(date +%Y%m%d_%H%M%S).jsonl"

bash nova5_tongji/scripts/run_tongji_controller.sh \
  --ekf 12d \
  --upright \
  --execute \
  --duration 60 \
  --observation-log "$observation_log" \
  --output-log /home/clover/ur_ws/ibvs_rm65/runtime/tongji_servoj_realtime.jsonl
```

该入口等待同济 EKF 收敛后才建立控制器；视觉、反馈、预测、关节、工作空间或跟随
误差任一失效都会调用 `Stop()`。默认控制周期为 50 Hz、相机到标记目标距离为
0.40 m、关节速度上限为 0.6 rad/s、关节加速度上限为 2 rad/s²，并使用实时平滑
的 `ServoJ`。12D 启动后使用 0.8 s 零斜率 ramp，进入 hold 前先按加速度限制减速。
当 EKF 检出目标线速度达到 `0.025 m/s` 或角速度达到 `0.12 rad/s` 时，控制器会在
`0.35 s` 内平滑切换到传送带高速档，Cartesian/关节速度上限提高到
`0.30 m/s / 1.1 rad/s`，关节加速度上限提高到 `6 rad/s²`；目标停止后自动回到
静止低速档。每帧日志中的 `motion_boost_scale=0..1` 和 `effective_limits`
可用于确认当前实际档位。
参数在
`nova5_tongji/field_runtime/config/tongji_direct_tracking.json`，Aimer 参数在
`nova5_tongji/field_runtime/config/tongji_marker_aimer.yaml`。

12D 模式沿用实时追踪的速度上限，并把 ESKF 的角速度用于正对姿态前馈。
`--observation-log` 每帧记录 PnP 位姿、创新、NIS、关节反馈和接受/拒绝状态；
该参数要求输出文件尚不存在，因此命令中使用时间戳文件名。

默认 12D 配置不预先区分“纯平移”和“平移+慢转”：ESKF 持续更新完整姿态，
静止时角速度自然收敛，纯平移时姿态保持不变，缓慢旋转则累积到姿态状态和角速度中。
`lock_static_target` 与 `lock_marker_orientation` 仅保留为诊断选项，默认均关闭。
角速度前馈在 `0.25–0.45 rad/s` 之间使用连续平滑启用，不再通过单一阈值硬开关，
避免平移过程中偶发角速度估计造成关节速度和方向突变。
正常追踪的关节加速度变化率限制为静止档 `12 rad/s³`、运动档 `30 rad/s³`；
目标误差跨零或关节限幅变化时，加速度不会瞬间从正最大值跳到负最大值。
当雅可比最小奇异值从 `0.18` 降到 `0.03` 时，速度上限连续减小到零；不会等到
硬奇异阈值才突然停机。日志字段 `singularity_speed_scale` 显示当前降速比例。

`tracker.max_gap` 为 `0.25 s`，只用于启动阶段容忍 ROS 位姿消息偶发延迟并累计
EKF 收敛样本；控制器运行后仍使用更严格的 `max_visual_age_s=0.15 s`，视觉失效
会立即停止追踪。

`target_loss_return.enabled=true` 时，控制器会在启动时记录当时的关节位姿。若
启动阶段始终没有可用目标，或运行中出现 `stale vision`、目标估计不可用、预测
超时、Jacobian 奇异、关节限位或工作空间保护，则使用五次平滑 ServoJ 轨迹
回到该启动位姿。轨迹起点匹配故障瞬间的实际关节速度，以不超过 `0.6 rad/s²`
平滑制动，回位巡航速度不超过 `0.2 rad/s`，并在最终报告中写入
`target_loss_return.triggered`、`trigger_reason` 和回位结果。伺服拒绝或反馈
失效时不继续盲发回位命令。
回位轨迹会匹配故障瞬间的实际关节速度，先零加速度平滑制动，再转向初始位姿，
避免高速追踪突然停止造成关节冲击声。
视觉中断超过 `0.06 s` 后，控制器不会立刻停止 ServoJ 流，而会继续以 50 Hz
发送升余弦速度曲线，加速度从零平滑增加到最大 `1.2 rad/s²` 后再回到零，把
命令速度和实测速度都降到 `0.02 rad/s` 以下，再启动回位；减速最长持续 `4 s`。
目标估计回调中的速度越界、预测失败等错误也先进入这条减速路径，不会绕过减速
直接切入回位。
达到零速后还会连续保持当前位置 `0.5 s`，让机械结构和反馈稳定后再反向回位。
进入正常视觉 hold 时同样限制为 `0.5 rad/s²` 缓慢减速，并要求命令速度和实测
关节速度都低于 `0.02 rad/s` 后才锁存静止。

`closed_loop.enabled=true` 时，回位成功后会清空旧 EKF 状态并等待新的连续
8 帧目标，然后自动建立新的控制器并继续追踪，不需要重启进程。
重新捕获时还要求目标基座坐标回到首轮起始位置 `0.15 m` 范围内，避免目标在
远处重新出现后机械臂直接追过去造成跑飞；超范围时保持初始位姿继续等待。
`wait_for_target_timeout_s=0` 表示目标暂时不在画面中时持续等待；
`--duration` 到期或手动 `Ctrl+C` 才会结束。最终报告中的
`closed_loop.cycles_started`、`closed_loop.recovery_events` 和
`target_loss_return.events` 会记录每次停机和重新捕获过程。
正常到达 `--duration` 或运行中手动 `Ctrl+C` 只停止，不自动回位。

传送带长期运行时使用无限时长；进程只在手动 `Ctrl+C` 时退出：

```bash
bash nova5_tongji/scripts/run_tongji_controller.sh \
  --ekf 12d \
  --upright \
  --execute \
  --duration 0 \
  --observation-log /home/clover/ur_ws/ibvs_rm65/runtime/tongji_12d_observations.jsonl \
  --output-log /home/clover/ur_ws/ibvs_rm65/runtime/tongji_12d_realtime.jsonl
```

只检查配置、不连接机械臂：

```bash
bash nova5_tongji/scripts/run_tongji_controller.sh
```

启动前确认没有 ROS Dobot 驱动占用端口：

```bash
pgrep -af 'dobot_bringup_v4_node|nova5_low_speed_executor.py'
```

正常应没有对应进程。停止时任一终端按 `Ctrl+C`；控制器退出前会调用 `Stop()`，
独立硬件急停始终保留。

`Aimer` 只提供指向，不负责机械臂跟随速度；因此 200 ms 回正仍必须实机测量，
不能由编译或离线测试宣称已达到。首次验证应从小偏移开始，并保留独立硬件急停。

2026-11-17 实机短测：服务周期 `50.078 Hz`、0 次超期，视觉无断流。7.7 cm /
20.3° 初始误差的关节速度伺服首轮用了约 2.64 s 进入 1 cm / 1.1°，未达到
200 ms。随后改为每周期最多 4 mm / 1.7° 的 `ServoP` 小步；静态保持测得
`49.979 Hz`、0 次超期。`ServoP` 的位移回中尚未在安全姿态下完成验收。
控制器现在按 Nova5 实时雅可比最小奇异值保护，而不是使用错误的固定 J5=90° 判断。

```bash
bash nova5_tongji/scripts/run_servo.sh --ros-args \
  -p config:="$PWD/nova5_tongji/site.local.json"
```

启动后空闲，不自动清错或使能。沿用 `/predictive_pbvs/start` 和 `/predictive_pbvs/stop`
服务，保留既有标定、时间戳、状态新鲜度、跟随误差和空间约束检查。
不要同时运行旧追踪程序或另一个 ServoJ 发布者。

### 固定初始位姿的重复测试

本次抖动验收的初始关节位姿记录在：

```text
/home/clover/ur_ws/ibvs_rm65/runtime/tongji_pose_baseline_20261118.json
```

只在当前位姿检查回位目标，不运动：

```bash
cd /home/clover/Desktop/ur5-pbvs-mujoco-simulation
.venv/bin/python nova5_tongji/scripts/restore_tongji_pose.py \
  --baseline /home/clover/ur_ws/ibvs_rm65/runtime/tongji_pose_baseline_20261118.json
```

以不超过 `0.2 rad/s` 的 ServoJ 运动回到基线，并确认反馈误差小于 `0.5°`：

```bash
.venv/bin/python nova5_tongji/scripts/restore_tongji_pose.py \
  --baseline /home/clover/ur_ws/ibvs_rm65/runtime/tongji_pose_baseline_20261118.json \
  --output /home/clover/ur_ws/ibvs_rm65/runtime/tongji_restore.json \
  --execute
```

连续测试时，每轮开始前和结束后都回到同一初始位姿。视觉短暂丢失导致的提前退出
会自动回位并重试，只把跑满时长的尝试作为有效轮次：

```bash
bash nova5_tongji/scripts/run_tongji_repeat_trials.sh \
  --execute \
  --trials 3 \
  --duration 10 \
  --attempts-per-trial 3 \
  --output-dir /home/clover/ur_ws/ibvs_rm65/runtime/tongji_repeat_fixed_start
```

2026-11-18 的三轮固定起点验收全部通过：最后 2 秒 `motion_hold=100%`，关节目标
增量和关节速度均为 0；TCP 位置峰峰值不超过 `0.014 mm`。三次测试后的回位最大
关节误差均约 `0.00007 rad`。原始日志和汇总见
`/home/clover/ur_ws/ibvs_rm65/runtime/tongji_repeat_each_from_baseline_final/`。

## 验证与限制

测试覆盖：实际原版 C++ EKF、相机运动解耦、未知线速度、未来像素、角度跨 ±π、
异常拒绝/时间戳、无重复外推、Nova5 URDF 雅可比、PBVS 适配限幅、QR/ArUco 检测和 HTTP JPEG。
合成验证不是独立实测真值。TCP 接口已在 Orin 上进行并行估计和小幅实机试验；
动态预测收益及 200 ms 回正目标尚未验收。ROS 入口的现场标定和接口仍需单独验收。

许可证：原仓库 MIT；同济子模块保留其 MIT LICENSE 和版权声明。
来源：[开源文章](https://bbs.robomaster.com/article/803315?source=4)。
