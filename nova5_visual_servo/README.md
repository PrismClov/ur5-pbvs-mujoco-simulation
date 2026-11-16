# Nova5 + D435i 预测前馈视觉伺服

这是完整的 ROS2 Python 包源码，包括视觉、模型、控制、驱动客户端、配置、启动、离线仿真和测试。不是已经实机验收的即插即用成品。默认不使能、不回初始位、不发送运动；现场参数和反馈时间语义验证后才能启动闭环。

## 依据与版本

- 结构参考 [RealMan visual-servo](https://github.com/RealManRobot/visual-servo/blob/ce8ebcb0be66203a22599e23ee8f07b5a08dbf30/src/interface.py)：视觉 → 雅可比 → 关节积分 → 驱动。未复制其代码；其 RM65、三维经验控制、200 Hz 设置不适用于此项目。
- [越疆官方 ROS2 V4](https://github.com/Dobot-Arm/DOBOT_6Axis_ROS2_V4/tree/7807ef8fb31e8f86a76cd474a4a9cddffef6eca9)：本次查询 main HEAD 为 `7807ef8`，README 标注 V4.6.5、Ubuntu 22.04 / ROS2 Humble，支持 Nova5。不是声称该版本适配所有 Nova5 固件。
- [RealSense ROS2](https://github.com/realsenseai/realsense-ros/tree/960306819f45006ca3f903aa046bed32c509679a)：锁定本次 ros2-master HEAD；需按该提交文档安装兼容 librealsense，未验证和本机 Humble 的联合编译。
- 官方 URDF 与参考 action server 保留原版权，许可证见 `third_party/DOBOT_LICENSE`。URDF 的 meshes 未打包，当前仅用于 FK/Jacobian，不用于 RViz；完整模型由上游提供。

## 预测怎么做

1. 曝光时刻的关节历史插值得到 q_visual，计算 `T_BQ = FK(q_visual) T_EC T_CQ`，消除相机自身运动。
2. 新帧由 12 维左乘误差状态 ESKF 联合估计 p、v、R、ω，传播协方差、用位姿创新修正速度；重复/乱序帧不更新。
3. `t_exec = now + execution_lead`，位置 `p+v*h`，旋转 `Exp(ω*h)R`；默认额外 lead 为 0，但补偿帧龄。
4. 机器人同样预测到 t_exec：`q+qd*(t_exec-t_feedback)`。这只是短时运动学近似，没有辨识执行器动态，没有使用 255 ms。
5. 预测位姿形成六维 PBVS 误差。前馈为 `[v_Q+ω_Q×r; ω_Q]`，r 是目标到期望末端的基座系偏置。
6. P 与前馈合成 → 加权 DLS → 速度/加速度限制 → 梯形积分 → 绝对 ServoJ。

匀速模型在转向/停止时会短暂预测错误；时域上限、异常观测和过期检查会限制使用范围，但不是稳定性证明。更多公式见 `控制原理.txt`。

## 工程结构

```text
nova5_visual_servo/
  core.py        前馈、PBVS、限制器与锁存
  target_pose_ekf.py  12 维 ESKF、协方差、创新门限和预测
  robot.py       官方 URDF FK/几何雅可比
  history.py     曝光时刻关节插值，不越界外插
  vision.py      D435i 彩色图像 + CameraInfo + QR PnP
  interface.py   ROS2 状态机、ServoJ/Stop、状态监督、日志
  simulate.py    无硬件离线闭环
models/          官方 Nova5 URDF
config/          现场配置模板
launch/          ROS2 启动
tests/           控制与模型回归
dependencies.repos 固定上游版本
```

## 离线运行（不需要 ROS）

在仓库根目录：

```bash
python -m pip install numpy
python -m unittest discover -s tests -v
python -m nova5_visual_servo.simulate --output ff.csv
python -m nova5_visual_servo.simulate --no-ff --output p.csv
```

仿真是理想笛卡尔执行器，模型测试另用官方 Nova5 URDF。不能将其结果说成 Nova5 动力学、像素回正或现场性能。

## ROS2 构建与相机

目标平台 Ubuntu 22.04 / ROS2 Humble，先安装 ROS2、rosdep、colcon、vcstool。将本仓库放到工作空间 src 下：

```bash
cd ~/servo_ws
vcs import src < src/nova5_visual_servo/dependencies.repos
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
export DOBOT_TYPE=nova5
export IP_address=192.168.5.1
ros2 launch dobot_bringup_v4 dobot_bringup_ros2.launch.py
```

另一个终端启动 D435i 官方驱动：

```bash
ros2 launch realsense2_camera rs_launch.py enable_color:=true enable_depth:=true align_depth.enable:=true
ros2 launch nova5_visual_servo servo.launch.py
```

本版位姿以已知尺寸二维码 PnP 为主，不使用深度或 IMU参与控制。二维码内容默认 `nova5-target`、物理边长默认 0.1 m，必须改成实际尺寸；用 `ros2 run nova5_visual_servo vision --ros-args -p side_m:=0.08 -p target_text:=你的内容` 可单独启动视觉。检测顺序为左上、右上、右下、左下，二维码系 x 向右、y 向下。

相机图像/内参默认话题 `/camera/camera/color/image_raw` 和 `/camera/camera/color/camera_info`，可 ROS remap。配置 `optical_frame` 必须与实际 Image header 一致。相机曝光时间与机器人采样时间必须完成同钟映射。

## 现场配置与反馈前置条件

`config/site.json` **全部设备边界、手眼、工作空间、ServoJ 参数是模板，不是实机推荐值**。T_EC 当前单位阵只是占位；必须填实际标定。T_QC_des 的 -0.5 m 仅针对本二维码坐标约定。URDF 是名义模型，需对照控制器 FK 验证工具/基座坐标和精度。

官方 main.cpp 默认 JointStatePublishRate=10 Hz；发布 header 是发布时刻，断连可继续发布旧值。单纯调到 50 Hz 不能解决源样本时刻/重复帧问题。本包不把它伪装成已经修好的反馈链：

- 实机前需使驱动发布经过校验的新反馈采样（建议至少 50 Hz）、可靠采样时间戳和断连状态；确认不会用新 header 包装旧样本。二进制接收器是否正确处理分片仍需专项验证。
- 完成该工作后才可设置 `feedback_sample_time_verified=true`。默认 false 会拒绝启动。
- 初始静止、手眼、限位、速度/加速度、跟随漂移和已验证停止流程完成后设置 `site_verified=true`。这两个字段是验收记录，不是用于跳过检查的开关。
- joint_states.velocity 不存在时，本包从相邻关节位置差分估计速度；噪声和采样延迟必须测量。正式模型可注入 core 的 predict_robot。
- 工作空间目前是末端包围盒加关节边界，**没有整臂碰撞检测或制动轨迹规划**；现场必须由独立控制器/监督提供。不能据此保证无碰撞。

配置可用 `ros2 run nova5_visual_servo servo --ros-args -p config:=/绝对路径/site.json` 指定。机器人由操作者按原流程使能，本节点从不自动 EnableRobot。

```bash
ros2 service call /predictive_pbvs/start std_srvs/srv/Trigger '{}'
ros2 service call /predictive_pbvs/stop std_srvs/srv/Trigger '{}'
ros2 bag record /visual_servo/log /visual_servo/target /joint_states_robot /dobot_msgs_v4/msg/RobotStatus
```

ServoJ 使用官方 `dobot_msgs_v4/srv/ServoJ` 的 a..f（度）、param_value；停止使用 Stop。服务名按官方单机器人模式，可通过 ROS remap 适配命名空间。只允许一个在途 ServoJ，下一周期仍未完成则故障停止，不无界排队。若现场服务往返不能满足 20 ms，不能声称实现 50 Hz，需优化官方发送链并重测。Stop 也依赖通信，应有独立硬件急停；停机服务返回只表示请求结果，不证明物理已停。

故障锁存后需要调查并重启节点；不会自动恢复。ROS 节点单线程顺序处理状态，视觉 PnP 在另一个进程。

## 验证边界

当前 Windows 环境可执行 NumPy 单元测试、官方模型数值差分和合成闭环；未安装 ROS2、OpenCV/SciPy，未做 ROS 编译、D435i 图像测试或机器人测试。PnP 重投影门限与运动跳变门限已实现，但平面 PnP 歧义仍需实际姿态/连续性验收。

本工程是可继续集成的完整包源码，**反馈驱动时间戳改造、现场标定、碰撞/制动监督和实机验证尚未完成**。不要把离线通过写成 300 ms 回正已实现。验收按固定目标纯 P、动态 FF、最后预测的顺序，每次只变一个因素。


## ESKF 更新说明

现已替换原差分低通 TargetTracker。用户附件只有设计说明和外链，未包含可直接导入的模块源码；本实现按该设计重新编写，不沿用附件宣称的 14 项测试或性能数字。

状态误差顺序为 [dp,dv,dtheta_base,domega_base]；采用左乘姿态误差、base 系角速度。Joseph 协方差更新后用 SO(3) 左雅可比重置。连续白加速度过程噪声的平移块精确离散；旋转块用 8 点 Gauss 积分。均值仍为恒速模型，并非恒加速度预测。

config/site.json 中四个主要量为 position_std_m、orientation_std_rad、linear_accel_psd_m2_s3、angular_accel_psd_rad2_s3；示例值需用实际 PnP 与手眼误差日志标定，velocity_tau 已移除。

predict_state(t,max_horizon) 返回预测 T、速度、12x12 covariance、位置/姿态最大标准差、horizon_s、usable 和 reason，且不改变主状态。predict 保留原返回 4x4 位姿的接口，不可用则抛 Fault，由控制层停止。未达到两次接受观测、连续拒绝达到门限、预测不确定度过大时不可用。新观测 NIS 超限返回 False，不刷新最后接受时刻；超过 max_gap 必须停止并重建实例。单次漏检可在视觉有效期及协方差门限内预测；绝不无限预测。

此更新没有改变机械臂侧 q+qd*h 的运动学预测，也没有辨识硬件延迟。关节反馈、时间映射、碰撞监督和实机验收限制仍然有效。
