"""ROS2 orchestration: visual pose -> PBVS -> ServoJ, RealMan-style short chain."""
import json
import time
from pathlib import Path
import numpy as np
from .core import Config, Controller, TargetTracker, Fault
from .robot import URDFRobot
from .history import JointHistory


def main():
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import String
    from std_srvs.srv import Trigger
    from dobot_msgs_v4.msg import RobotStatus
    from dobot_msgs_v4.srv import ServoJ, Stop
    from scipy.spatial.transform import Rotation
    from ament_index_python.packages import get_package_share_directory

    class Servo(Node):
        def __init__(self):
            super().__init__('predictive_pbvs')
            share = Path(get_package_share_directory('nova5_visual_servo'))
            self.declare_parameter('config', str(share/'config/site.json'))
            self.cfg = json.loads(Path(self.get_parameter('config').value).read_text())
            self.settings = Config(**self.cfg['control'])
            self.model = URDFRobot(share/'models/nova5_robot.urdf')
            self.history = JointHistory(self.cfg['feedback_max_gap'])
            self.tracker = TargetTracker(**self.cfg['tracker'])
            self.T_EC = np.array(self.cfg['T_EC'])
            self.pending = None
            self.ctrl = None
            self.faulted = False
            self.status = None
            self.future = None
            self.last_ros = None
            self.servo = self.create_client(ServoJ, '/dobot_bringup_ros2/srv/ServoJ')
            self.stop_client = self.create_client(Stop, '/dobot_bringup_ros2/srv/Stop')
            self.logs = self.create_publisher(String, '/visual_servo/log', 10)
            self.create_subscription(JointState, '/joint_states_robot', self.feedback, 10)
            self.create_subscription(RobotStatus, '/dobot_msgs_v4/msg/RobotStatus', self.robot_status, 1)
            self.create_subscription(PoseStamped, '/visual_servo/target', self.target, 1)
            self.create_service(Trigger, '~/start', self.start)
            self.create_service(Trigger, '~/stop', self.stop)
            self.create_timer(.02, self.tick)
            self.get_logger().info('Idle. No automatic enable or motion. Start service requires verified site config.')

        def now(self):
            return self.get_clock().now().nanoseconds/1e9

        def fail(self, reason):
            self.faulted = True
            self.ctrl = None
            self.get_logger().error(str(reason))
            if self.stop_client.service_is_ready():
                future = self.stop_client.call_async(Stop.Request())
                future.add_done_callback(self.stop_result)
            else:
                self.get_logger().fatal('Stop service unavailable; use independent hardware stop')

        def stop_result(self, f):
            try:
                if f.result().res != 0:
                    raise Fault('Stop rejected')
            except Exception as e:
                self.get_logger().fatal(f'Stop failed: {e}; use hardware stop')

        def robot_status(self, m):
            self.status = (time.monotonic(), m.is_connected, m.is_enable)
            if self.ctrl is not None and not (m.is_connected and m.is_enable):
                self.fail('robot disconnected or disabled')

        def feedback(self, m):
            try:
                q = [m.position[m.name.index(n)] for n in self.model.names]
                stamp = m.header.stamp.sec+m.header.stamp.nanosec/1e9
                # Upstream default is publication time. Site must provide calibrated sample stamps.
                self.history.add(stamp, q)
            except Exception as e:
                if self.ctrl is not None:
                    self.fail(e)

        def target(self, m):
            self.pending = m

        def consume_target(self):
            if self.pending is None:
                return
            m = self.pending
            t = m.header.stamp.sec+m.header.stamp.nanosec/1e9
            if m.header.frame_id != self.cfg['optical_frame']:
                raise Fault('unexpected camera optical frame')
            if not self.history.samples or t > self.history.samples[-1][0]:
                return  # wait for bracketing feedback, limited by visual age on tick
            self.pending = None
            q = self.history.at(t)
            T = np.eye(4)
            p, r = m.pose.position, m.pose.orientation
            T[:3, 3] = [p.x, p.y, p.z]
            T[:3, :3] = Rotation.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
            self.tracker.update(m.header.stamp.sec*1000000000+m.header.stamp.nanosec,
                                t, self.model.fk(q), self.T_EC, T)

        def workspace(self, q):
            p = self.model.fk(q)[:3, 3]
            return bool(np.all(p >= self.cfg['workspace_min']) and np.all(p <= self.cfg['workspace_max']))

        def start(self, req, res):
            try:
                if self.ctrl is not None or self.faulted:
                    raise Fault('already active or fault latched; restart after investigation')
                if not self.cfg['site_verified'] or not self.cfg['feedback_sample_time_verified']:
                    raise Fault('site calibration, limits and feedback source timestamps must be verified')
                if not self.servo.service_is_ready() or not self.stop_client.service_is_ready():
                    raise Fault('driver services unavailable')
                if self.status is None or time.monotonic()-self.status[0] > .1 or not all(self.status[1:]):
                    raise Fault('robot status unavailable/not enabled')
                self.consume_target()
                t, q, qd = self.history.latest()
                if np.max(abs(qd)) > .01:
                    raise Fault('start requires stationary robot')
                if self.tracker.t is None or self.now()-self.tracker.t > self.settings.max_visual_age:
                    raise Fault('fresh vision required')
                if not self.tracker.predict_state(self.now(), self.settings.max_prediction)['usable']:
                    raise Fault('ESKF requires accepted observations and bounded uncertainty')
                self.ctrl = Controller(self.settings, self.tracker, self.cfg['T_QC_des'], self.T_EC,
                    self.model.fk, self.model.jac, self.workspace, q, qd, self.now())
                res.success, res.message = True, 'started'
            except Exception as e:
                res.success, res.message = False, str(e)
            return res

        def stop(self, req, res):
            self.fail('operator stop')
            res.success, res.message = True, 'stop requested; check driver result and physical stop'
            return res

        def tick(self):
            try:
                now = self.now()
                if self.last_ros is not None and now <= self.last_ros:
                    raise Fault('ROS clock discontinuity')
                self.last_ros = now
                if self.faulted:
                    return
                self.consume_target()
                if self.ctrl is None:
                    return
                if self.status is None or time.monotonic()-self.status[0] > .1 or not all(self.status[1:]):
                    raise Fault('stale robot status')
                if self.future is not None:
                    if not self.future.done():
                        raise Fault('ServoJ did not complete before next cycle; no command queueing')
                    if self.future.result().res != 0:
                        raise Fault('ServoJ rejected')
                t, q, qd = self.history.latest()
                out = self.ctrl.tick(now, q, qd, t)
                req = ServoJ.Request()
                req.a, req.b, req.c, req.d, req.e, req.f = np.degrees(out['q_plan']).tolist()
                req.param_value = self.cfg['servoj_parameters']
                self.future = self.servo.call_async(req)
                msg = String()
                msg.data = json.dumps(dict(time=now, q_actual=q.tolist(), **{
                    k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in out.items()}))
                self.logs.publish(msg)
            except Exception as e:
                if self.ctrl is not None:
                    self.fail(e)
                else:
                    self.get_logger().warning(str(e), throttle_duration_sec=2.)

    rclpy.init()
    node = Servo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.fail('shutdown requested')
        # Allow asynchronous stop result to arrive; hardware stop remains independent.
        end = time.monotonic()+1.
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=.05)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
