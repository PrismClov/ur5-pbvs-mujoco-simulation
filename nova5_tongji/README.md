# Nova5 × 同济 SuperPower：传送带标记跟踪

独立工程，复用 [同济 sp_vision_25](https://github.com/TongjiSuperPower/sp_vision_25)
原版 C++ `tools::ExtendedKalmanFilter`，通过外层适配连接二维码/ArUco、Nova5
正运动学、手眼标定、PBVS 和 ServoJ。仓库原有仿真与 `nova5_visual_servo` 文件不变。

**当前状态：原版 EKF 已在 Windows 实际编译并接入测试；ROS2 节点、预览和 ServoJ
适配入口已提供，尚未在 Orin 编译或完成实机闭环验收。不能据此宣称实现了 200 ms 回正。**

## 原版保留范围

上游作为 Git 子模块完整保留，固定提交：
`bd9f5e798fa3c6dd3b483ae6627796afb41c608d`。本工程不修改其源码。
构建直接编译上游 `tools/extended_kalman_filter.cpp`，不是另写一份 Python KF。

| 部分 | 处理 |
| --- | --- |
| EKF 状态传播、增益和 Joseph 协方差更新 | 直接使用同济原版实现 |
| 原装甲板识别、整车旋转模型、云台规划器 | 保存在子模块；二维码执行路径不调用 |
| 单标记观测/运动模型 | 新增 `src/conveyor_adapter.cpp` |
| 相机、正运动学、手眼变换 | 外层适配，使用时间配对关节状态 |
| 六轴 PBVS、约束、ServoJ | 复用本仓库既有 Nova5 实现 |

原版 `Target` 使用旋转中心、半径、多块装甲板等状态，`update()` 中还有取最近三块板
的循环，因此不能简单设置“装甲板数=1”来处理二维码。原云台 yaw/pitch 规划器也不能直接
输出六轴关节位置。本工程保留原版供对照，明确增加必要适配，不声称整条自瞄链可原样运行。

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

可通过 ROS `-r 原话题:=现场话题` 重映射。旧项目的直接 V4L2/TCP 启动命令不能直接套用：
该版本的实机适配使用本仓库现有 ROS2/Dobot 驱动接口。

## 机械臂执行入口

```bash
bash nova5_tongji/scripts/run_servo.sh --ros-args \
  -p config:="$PWD/nova5_tongji/site.local.json"
```

启动后空闲，不自动清错或使能。沿用 `/predictive_pbvs/start` 和 `/predictive_pbvs/stop`
服务，保留既有标定、时间戳、状态新鲜度、跟随误差和空间约束检查。
不要同时运行旧追踪程序或另一个 ServoJ 发布者。

## 验证与限制

测试覆盖：实际原版 C++ EKF、相机运动解耦、未知线速度、未来像素、角度跨 ±π、
异常拒绝/时间戳、无重复外推、Nova5 URDF 雅可比、PBVS 适配限幅、QR/ArUco 检测和 HTTP JPEG。
合成验证不是独立实测真值。Orin 构建、ROS 话题对接、标定迁移、实机运动仍待现场验收。

许可证：原仓库 MIT；同济子模块保留其 MIT LICENSE 和版权声明。
来源：[开源文章](https://bbs.robomaster.com/article/803315?source=4)。
