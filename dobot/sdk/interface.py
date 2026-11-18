import re
import numpy as np
import time
import transforms3d as t3d

from .dobot_api import DobotApiFeedBack, DobotApiDashboard
from .ag95_gripper import AG95GripperDobot
from .do_gripper import DOGripperDobot

_EXTERNAL_GRIPPER_TYPES = ()


import threading


class FeedbackItem:
    def __init__(self):
        self.robotMode = -1     #
        self.robotCurrentCommandID = 0
        self.controllerTimestamp = None
        self.controllerRuntime = None
        self.MessageSize = -1
        self.DigitalInputs =-1
        self.DigitalOutputs = -1
        self.robotCurrentCommandID = -1
        self.QTarget = None
        self.QDTarget = None
        self.QDDTarget = None
        self.QActual = None
        self.QDActual = None
        self.ToolVectorActual = None
        self.ToolVectorTarget = None
        self.TCPSpeedActual = None
        self.TCPSpeedTarget = None
        self.ActualTCPForce = None
        self.MActual = None

class DobotInterface:
    def __init__(
        self, 
        dashboardPort = 29999,
        feedPortFour = 30004,
        ip = "192.168.5.1",
        speed = 20,
        t = 0.2,
        aheadtime = 50,
        gain = 500,
        gripper_type = None,
        gripper_force = 80,
        gripper_velocity = 500,
        gripper_slave_id = 1,
        gripper_parity = "N",
        gripper_do_index = 1,
        tool_idx = 0,
        damiao_gripper=None,
        gripper_dobot=None,
        joint_command_max_velocity: float = 2.0,
        joint_command_max_dt: float = 0.05,
        joint_command_smoothing_factor: float = 0.1,
        update_fps: float = 60.0,
    ):
        self.value_lock = threading.Lock()
        self.dashboardPort = dashboardPort
        self.feedPortFour = feedPortFour
        self.ip = ip
        self.speed = speed
        self.t = t
        self.aheadtime = aheadtime
        self.gain = gain
        self.gripper_force = gripper_force
        self.gripper_velocity = gripper_velocity
        self.gripper_do_index = gripper_do_index
        self.tool_idx = tool_idx

        self._joint_cmd_max_velocity = float(joint_command_max_velocity)
        self._joint_cmd_max_dt = float(joint_command_max_dt)
        self._joint_cmd_smoothing_factor = float(joint_command_smoothing_factor)
        self._joint_cmd_last_time: float | None = None
        self._joint_cmd_prev_position: np.ndarray | None = None
        self._joint_cmd_lock = threading.Lock()
        self.update_fps = float(update_fps)
        if self.update_fps <= 0:
            raise ValueError("update_fps must be positive")
        self._feedback_stop = threading.Event()
        self._feedback_thread: threading.Thread | None = None
        self.feedback_count = 0
        self.feedback_last_monotonic: float | None = None
        self.feedback_last_duration_s: float | None = None
        self.feedback_invalid_count = 0
        self.servo_send_count = 0
        self.servo_send_last_monotonic: float | None = None
        self.servo_send_last_duration_s: float | None = None
        self.servo_send_last_result = None

        self.dashboard = DobotApiDashboard(ip, dashboardPort)
        self.feedFour = DobotApiFeedBack(ip, feedPortFour)

        self.dashboard.ClearError()
        self.dashboard.Stop()
        if self.parseResultId(self.dashboard.EnableRobot())[0] != 0:
            print("使能失败: 检查29999端口是否被占用")
            raise ValueError("使能失败: 检查29999端口是否被占用或者机器人是否在TCP模式下")
        print("使能成功")

        self.dashboard.SpeedFactor(self.speed)
        print(f"全局速度比例设置成{self.speed}%")
        self.dashboard.VelJ(self.speed)
        print(f"Joint速度比例设置成{self.speed}%")
        self.dashboard.AccJ(self.speed)
        print(f"Joint加速度比例设置成{self.speed}%")

        # Initialize gripper based on type (optional)
        self.gripper = None
        if gripper_dobot is not None:
            self.gripper = gripper_dobot
            print("Using pre-initialized external gripper adapter")
        elif gripper_type is not None and gripper_type.strip():
            gripper_type_lower = gripper_type.lower().strip()
            if gripper_type_lower == "ag95":
                self.gripper = AG95GripperDobot(self.dashboard, slave_id=gripper_slave_id, parity=gripper_parity)
                # Set initial gripper parameters
                print(f"Setting gripper force: {self.gripper_force}, velocity: {self.gripper_velocity}")
                self.gripper.set_force(self.gripper_force)
                self.gripper.set_vel(self.gripper_velocity)
            elif gripper_type_lower == "pge":
                self.gripper = PGEGripperDobot(self.dashboard, slave_id=gripper_slave_id, parity=gripper_parity)
                # Set initial gripper parameters
                print(f"Setting gripper force: {self.gripper_force}, velocity: {self.gripper_velocity}")
                self.gripper.set_force(self.gripper_force)
                self.gripper.set_vel(self.gripper_velocity)
            elif gripper_type_lower == "do":
                self.gripper = DOGripperDobot(self.dashboard, do_index=gripper_do_index)
                # DO gripper doesn't support force/velocity control
                print(f"DO gripper initialized on DO{gripper_do_index}")
            elif gripper_type_lower == "damiao":
                from .damiao_gripper_dobot import DamiaoGripperDobot

                if damiao_gripper is None:
                    from leapcollect.robots.gripper.damiao import DamiaoGripperConfig

                    damiao_gripper = DamiaoGripperConfig()
                self.gripper = DamiaoGripperDobot.from_config(damiao_gripper)
                print(
                    f"Damiao gripper can_id=0x{damiao_gripper.can_id:02X} "
                    f"master_id=0x{damiao_gripper.master_id:02X} "
                    f"mode={damiao_gripper.control_mode} "
                    f"torque_limit_nm={damiao_gripper.torque_limit_nm}"
                )
            else:
                raise ValueError(
                    f"Unknown gripper type: {gripper_type}. "
                    "Supported types: 'ag95', 'pge', 'do', 'damiao'"
                )
        else:
            print("No gripper type specified - operating without gripper")

        self._gripper_last_pos: float = 1.0

        self.feedData = FeedbackItem()

        self._feedback_thread = threading.Thread(
            target=self.GetFeed,
            name="dobot-feedback",
            daemon=True,
        )
        self._feedback_thread.start()

    def GetFeed(self):
        """Continuously snapshot robot feedback at ``update_fps``.

        The source remains the robot's own feedback publication rate. This
        loop only controls how often the latest packet is copied into the
        process-local snapshot.
        """
        period = 1.0 / self.update_fps
        deadline = time.monotonic()
        while not self._feedback_stop.is_set():
            remaining = deadline - time.monotonic()
            if remaining > 0 and self._feedback_stop.wait(remaining):
                break
            s0 = time.monotonic()
            try:
                feedInfo = self.feedFour.feedBackData()
            except OSError as e:
                print(e)
                break
            except Exception as e:
                print(e)
                continue

            with self.value_lock:
                if feedInfo is not None:   
                    if hex((feedInfo['TestValue'][0])) == '0x123456789abcdef':
                        # 基础字段
                        self.feedData.MessageSize = feedInfo['len'][0]
                        self.feedData.robotMode = feedInfo['RobotMode'][0]
                        self.feedData.controllerTimestamp = int(
                            feedInfo['TimeStamp'][0])
                        self.feedData.controllerRuntime = int(
                            feedInfo['RunTime'][0])
                        self.feedData.DigitalInputs = feedInfo['DigitalInputs'][0]
                        self.feedData.DigitalOutputs = feedInfo['DigitalOutputs'][0]
                        self.feedData.robotCurrentCommandID = feedInfo['CurrentCommandId'][0]
                        self.feedData.QTarget = feedInfo['QTarget'][0]
                        self.feedData.QDTarget = feedInfo['QDTarget'][0]
                        self.feedData.QDDTarget = feedInfo['QDDTarget'][0]
                        self.feedData.QActual = feedInfo['QActual'][0]
                        self.feedData.QDActual = feedInfo['QDActual'][0]
                        self.feedData.ToolVectorActual = feedInfo['ToolVectorActual'][0]
                        self.feedData.ToolVectorTarget = feedInfo['ToolVectorTarget'][0]
                        self.feedData.TCPSpeedActual = feedInfo['TCPSpeedActual'][0]
                        self.feedData.TCPSpeedTarget = feedInfo['TCPSpeedTarget'][0]
                        self.feedData.ActualTCPForce = feedInfo['ActualTCPForce'][0]
                        self.feedData.MActual = feedInfo['MActual'][0]
                        self.feedback_count += 1
                        self.feedback_last_monotonic = time.monotonic()
                        self.feedback_last_duration_s = (
                            self.feedback_last_monotonic - s0)
                    else:
                        self.feedback_invalid_count += 1
            deadline += period
            now = time.monotonic()
            if deadline <= now:
                missed = int((now - deadline) / period) + 1
                deadline += missed * period

    def parseResultId(self, valueRecv):
        # 解析返回值，确保机器人在 TCP 控制模式
        if "Not Tcp" in valueRecv:
            print("Control Mode Is Not Tcp")
            return [1]
        return [int(num) for num in re.findall(r'-?\d+', valueRecv)] or [2]
    
    def get_end_pose(self):
        """Return current TCP pose as ``[x, y, z, rx, ry, rz]`` (metres, sxyz Euler radians)."""
        with self.value_lock:
            nums = self.feedData.ToolVectorActual
        if nums is None:
            return None
        end_pose_arr = np.array([float(x) for x in nums])
        translation = end_pose_arr[:3] / 1000
        euler_rad = np.deg2rad(end_pose_arr[3:])
        return np.concatenate([translation, euler_rad])

    def get_joint_position(self):
        with self.value_lock:
            nums = self.feedData.QActual
        if nums is not None:
            joints = [float(x) for x in nums]
            joints_rad = np.deg2rad(joints)
            return joints_rad.tolist()
        return None

    def feedback_snapshot(self):
        """Return an immutable-by-copy snapshot of the latest feedback."""
        with self.value_lock:
            joint_position_deg = (
                None if self.feedData.QActual is None
                else [float(x) for x in self.feedData.QActual])
            joint_target_deg = (
                None if self.feedData.QTarget is None
                else [float(x) for x in self.feedData.QTarget])
            return {
                "robot_mode": self.feedData.robotMode,
                "command_id": self.feedData.robotCurrentCommandID,
                "controller_timestamp": self.feedData.controllerTimestamp,
                "controller_runtime": self.feedData.controllerRuntime,
                "joint_position_deg": (
                    None if joint_position_deg is None else joint_position_deg.copy()),
                "joint_position_rad": (
                    None if joint_position_deg is None
                    else np.deg2rad(joint_position_deg).tolist()),
                "joint_target_deg": (
                    None if joint_target_deg is None else joint_target_deg.copy()),
                "joint_target_rad": (
                    None if joint_target_deg is None
                    else np.deg2rad(joint_target_deg).tolist()),
                "joint_velocity_target_deg_s": (
                    None if self.feedData.QDTarget is None
                    else [float(x) for x in self.feedData.QDTarget]),
                "joint_acceleration_target_deg_s2": (
                    None if self.feedData.QDDTarget is None
                    else [float(x) for x in self.feedData.QDDTarget]),
                "joint_velocity_actual_deg_s": (
                    None if self.feedData.QDActual is None
                    else [float(x) for x in self.feedData.QDActual]),
                "tool_vector_mm_deg": (
                    None if self.feedData.ToolVectorActual is None
                    else [float(x) for x in self.feedData.ToolVectorActual]),
                "tool_vector_target_mm_deg": (
                    None if self.feedData.ToolVectorTarget is None
                    else [float(x) for x in self.feedData.ToolVectorTarget]),
                "tcp_speed_actual": (
                    None if self.feedData.TCPSpeedActual is None
                    else [float(x) for x in self.feedData.TCPSpeedActual]),
                "tcp_speed_target": (
                    None if self.feedData.TCPSpeedTarget is None
                    else [float(x) for x in self.feedData.TCPSpeedTarget]),
                "monotonic": self.feedback_last_monotonic,
                "feedback_count": self.feedback_count,
                "loop_period_s": self.feedback_last_duration_s,
            }
    
    def get_gripper_position(self):
        """Return gripper position (normalized 0--1).

        Modbus grippers return the last commanded position; Damiao reads
        hardware via ``get_normalized_position()``.
        """
        if self.gripper is None:
            return None
        if isinstance(self.gripper, _EXTERNAL_GRIPPER_TYPES):
            return self.gripper.get_normalized_position()
        return float(self._gripper_last_pos)

    def set_joints(self, goal_joint_rad, t=None, aheadtime=None, gain=None):
        goal_joint_rad = np.asarray(goal_joint_rad, dtype=np.float64).reshape(-1)
        if goal_joint_rad.size != 6:
            raise AssertionError(f"Dobot should input 6 joints, current [{goal_joint_rad.size}]")

        now_sec = time.time()
        max_vel = max(self._joint_cmd_max_velocity, 1e-6)
        max_dt = max(self._joint_cmd_max_dt, 1e-6)
        smoothing_factor = float(np.clip(self._joint_cmd_smoothing_factor, 0.0, 1.0))

        with self._joint_cmd_lock:
            current_position = self.get_joint_position()
            if current_position is not None and len(current_position) >= 6:
                current_position_arr = np.asarray(current_position[:6], dtype=np.float64)
            else:
                current_position_arr = goal_joint_rad.copy()

            if self._joint_cmd_last_time is None:
                dt_raw = 0.02
            else:
                dt_raw = max(now_sec - self._joint_cmd_last_time, 1e-6)
            dt_capped = min(dt_raw, max_dt)

            if self._joint_cmd_prev_position is not None and dt_raw <= 0.5:
                reference = self._joint_cmd_prev_position
            else:
                reference = current_position_arr

            smoothed_target = (1.0 - smoothing_factor) * goal_joint_rad + smoothing_factor * reference
            joint_velocity = np.clip((smoothed_target - reference) / dt_capped, -max_vel, max_vel)
            limited_position = reference + joint_velocity * dt_capped

            self._joint_cmd_prev_position = limited_position
            self._joint_cmd_last_time = now_sec

        goal_joint_deg = np.rad2deg(limited_position)
        j1, j2, j3, j4, j5, j6 = goal_joint_deg.tolist()
        send_started = time.monotonic()
        result = self.dashboard.ServoJ(
            j1,
            j2,
            j3,
            j4,
            j5,
            j6,
            self.t if t is None else t,
            self.aheadtime if aheadtime is None else aheadtime,
            self.gain if gain is None else gain,
        )
        send_finished = time.monotonic()
        self.servo_send_count += 1
        self.servo_send_last_monotonic = send_finished
        self.servo_send_last_duration_s = send_finished - send_started
        self.servo_send_last_result = result
        return limited_position.tolist()

    def set_pose(self, pose_quat, position_tolerance=0.005, orientation_tolerance=0.01, timeout=10.0):
        """
        Move to pose and wait for end effector to reach the target pose.
        
        Args:
            pose_quat: [x, y, z, qw, qx, qy, qz] - target pose in meters and quaternion
            position_tolerance: Maximum allowed position error in meters (default: 2mm)
            orientation_tolerance: Maximum allowed quaternion distance (default: 0.01)
            timeout: Maximum time to wait in seconds (default: 30s)
        
        Returns:
            True if pose reached, False if timeout
        """
        pose_quat = np.array(pose_quat)

        # Send movement command
        self.move_pose(pose_quat, user=0, tool=self.tool_idx)
        
        # Wait for end effector to reach target pose
        start_time = time.time()
        pos_error = float('inf')
        quat_distance = float('inf')
        time.sleep(0.2)  # Small sleep to avoid excessive CPU usage
        
        while time.time() - start_time < timeout:
            if self.feedData.robotMode == 5:
                return True
            
            time.sleep(0.01)  # Small sleep to avoid excessive CPU usage
        
        print(f"Warning: set_pose timeout after {timeout}s. Position error: {pos_error:.4f}m, Orientation error: {quat_distance:.4f}")
        return False

    def move_pose(self, pose_quat, user=-1, tool=1, a=-1, v=-1, speed=-1, cp=-1, r=-1):
        """
        Move to pose using MovL (linear motion) with configurable parameters.
        This is useful for initial movements where you want to control speed, acceleration, and blending.
        
        Args:
            pose_quat: [x, y, z, qw, qx, qy, qz] - target pose in meters and quaternion
            user: User coordinate system index
            tool: Tool coordinate system index
            a: Acceleration rate (0-100], percentage
            v: Velocity rate (0-100], percentage (mutually exclusive with speed)
            speed: Target speed in mm/s (mutually exclusive with v)
            cp: Continuous path rate [0-100] (mutually exclusive with r)
            r: Continuous path radius in mm (mutually exclusive with cp)
        
        Returns:
            Result from MovL command
        """
        # convert from meter to mm
        xyz = np.array(pose_quat[:3]) * 1000

        qw, qx, qy, qz = pose_quat[3:]
        rpy = t3d.euler.quat2euler([qw, qx, qy, qz], axes='sxyz')

        # convert from rad to deg
        rpy = np.degrees(rpy)
        x, y, z = xyz
        rx, ry, rz = rpy
        
        # Use MovL with pose mode (coordinateMode=0)
        return self.dashboard.MovL(x, y, z, rx, ry, rz, coordinateMode=0, 
                                   user=user, tool=tool, a=a, v=v, speed=speed, cp=cp, r=r)

    def set_gripper(self, gripper_pos):
        """Set gripper position in ``[0, 1]`` (closed to open). Maps internally to hardware 0--1000."""
        if self.gripper is None:
            print("Warning: No gripper attached - set_gripper call ignored")
            return

        gripper_pos = float(np.clip(gripper_pos, a_min=0.0, a_max=1.0))
        if isinstance(self.gripper, _EXTERNAL_GRIPPER_TYPES):
            self.gripper.set_pos(gripper_pos)
        else:
            self.gripper.set_pos(int(round(gripper_pos * 1000.0)))
        self._gripper_last_pos = gripper_pos
        # self.gripper.set_force(self.gripper_force)
        # self.gripper.set_vel(self.gripper_velocity)

    def disconnect(self):
        self._feedback_stop.set()
        thread = self._feedback_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._feedback_thread = None
        if isinstance(self.gripper, _EXTERNAL_GRIPPER_TYPES):
            if getattr(self.gripper, "_owns_bus", True):
                self.gripper.close()
            self.gripper = None
        self.dashboard.close()
        self.feedFour.close()
        # del self.dashboard
        # del self.feedFour

    # ========== Force Control Methods ==========
    
    def get_current_force(self):
        """
        Read current force from dobot feedback.
        Returns: [fx, fy, fz, frx, fry, frz] in N, or None if error
        """
        force = self.feedData.ActualTCPForce
        if force is not None:
            return np.array(force, dtype=np.float64)  # [fx, fy, fz, frx, fry, frz]
        return None
    
    def get_force(self):
        """
        Common interface method to get force/torque data.
        Returns: [fx, fy, fz, frx, fry, frz] in N (forces) and N·m (torques), or None if error
        This is a wrapper around get_current_force() for common interface compatibility.
        """
        return self.get_current_force()

    def get_joint_torque(self):
        """
        Read current joint torque from Dobot feedback (MActual).
        Returns: [t1, t2, t3, t4, t5, t6] in N·m, or None if unavailable
        """
        torque = self.feedData.MActual
        if torque is not None:
            return np.array(torque, dtype=np.float64)
        return None

    def disable_force_control(self):
        """Disable force control mode."""
        return self.dashboard.FCOff()
