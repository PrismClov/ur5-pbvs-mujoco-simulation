"""Timestamped marker observations and image predictions. No motion API."""
import json
from pathlib import Path
import numpy as np
from .vision import detect_pose
from .tracker import TongjiTargetTracker
from .preview import Preview
from nova5_visual_servo.core import pose
from nova5_visual_servo.robot import URDFRobot
from nova5_visual_servo.history import JointHistory

def main():
    import cv2
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image,CameraInfo,JointState
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import String
    from cv_bridge import CvBridge
    from scipy.spatial.transform import Rotation
    from ament_index_python.packages import get_package_share_directory
    class Vision(Node):
        def __init__(self):
            super().__init__('tongji_marker_vision')
            root=Path(__file__).resolve().parents[2]
            for name,value in dict(config=str(root/'site.example.json'),kind='aruco',
                    marker_id=0,target_text='nova5-target',side_m=.1,preview_host='127.0.0.1',preview_port=8765).items():
                self.declare_parameter(name,value)
            self.cfg=json.loads(Path(self.get_parameter('config').value).read_text())
            self.T_EC=pose(self.cfg['T_EC'])
            self.calibrated=bool(self.cfg['site_verified'])
            share=Path(get_package_share_directory('nova5_visual_servo'))
            self.robot=URDFRobot(share/'models/nova5_robot.urdf')
            self.history=JointHistory(self.cfg['feedback_max_gap'])
            self.tracker=TongjiTargetTracker(**self.cfg['tracker'])
            self.preview=Preview(self.get_parameter('preview_host').value,self.get_parameter('preview_port').value)
            self.bridge=CvBridge();self.info=None;self.pending=None;self.seq=0
            self.target=self.create_publisher(PoseStamped,'/visual_servo/target',1)
            self.diagnostics=self.create_publisher(String,'/tongji/prediction',10)
            self.create_subscription(CameraInfo,'/camera/camera/color/camera_info',self.camera_info,qos_profile_sensor_data)
            self.create_subscription(Image,'/camera/camera/color/image_raw',self.image,qos_profile_sensor_data)
            self.create_subscription(JointState,'/joint_states_robot',self.joints,10)
            self.create_timer(.01,self.process)

        def camera_info(self,m):
            self.info=(np.array(m.k).reshape(3,3),np.array(m.d),m.header.frame_id) if m.distortion_model in ('plumb_bob','rational_polynomial') else None

        def joints(self,m):
            try:self.history.add(m.header.stamp.sec+m.header.stamp.nanosec/1e9,[m.position[m.name.index(n)] for n in self.robot.names])
            except (ValueError,IndexError):pass

        def image(self,m):
            if self.info is None:return
            K,D,frame=self.info
            if m.header.frame_id!=frame or frame!=self.cfg['optical_frame']:return
            image=self.bridge.imgmsg_to_cv2(m,'bgr8')
            result=detect_pose(image,K,D,self.get_parameter('side_m').value,self.get_parameter('kind').value,
                               self.get_parameter('marker_id').value,self.get_parameter('target_text').value)
            if result is None:
                self.pending=None
                cv2.putText(image,'NO TARGET - no prediction/control observation',(12,30),0,.6,(0,180,255),2)
                self.preview.publish(image);return
            T,pixels,error=result
            out=PoseStamped();out.header=m.header
            out.pose.position.x,out.pose.position.y,out.pose.position.z=T[:3,3].tolist()
            out.pose.orientation.x,out.pose.orientation.y,out.pose.orientation.z,out.pose.orientation.w=Rotation.from_matrix(T[:3,:3]).as_quat().tolist()
            self.target.publish(out)
            cv2.polylines(image,[pixels.astype(np.int32)],True,(0,255,0),2)
            self.seq+=1
            stamp=m.header.stamp.sec+m.header.stamp.nanosec/1e9
            self.pending=(self.seq,stamp,T,image,K,D)
            if not self.calibrated:
                cv2.putText(image,'Calibration unverified: detection preview only',(12,30),0,.6,(0,180,255),2)
                self.preview.publish(image);self.pending=None
            else:
                waiting=image.copy()
                cv2.putText(waiting,'Detected - waiting for paired joint feedback',(12,30),0,.6,(0,180,255),2)
                self.preview.publish(waiting)

        def process(self):
            if self.pending is None:return
            seq,t,T,image,K,D=self.pending
            now=self.get_clock().now().nanoseconds/1e9
            if not 0<=now-t<=.15:
                self.pending=None
                cv2.putText(image,'Image expired / clock mismatch',(12,30),0,.6,(0,180,255),2)
                self.preview.publish(image);return
            try:q=self.history.at(t)
            except ValueError:return # wait for bracketing feedback, never substitute current q
            self.pending=None
            row=dict(sequence=seq,image_time=t,clock='ROS image + joint stamps',prediction_clock_verified=bool(self.cfg['feedback_sample_time_verified']),pixels={})
            try:
                self.tracker.update(seq,t,self.robot.fk(q),self.T_EC,T)
                row.update(status=self.tracker.last_status,speed_m_s=float(np.linalg.norm(self.tracker.v)))
                _,_,qd=self.history.latest()
                for h,color in [(0.,(255,255,0)),(.05,(0,255,255)),(.1,(255,0,255))]:
                    if self.tracker.last_status not in ('initialized','updated'):break
                    dq=qd*h
                    if np.linalg.norm(dq)>.05 or np.max(abs(qd))>2:continue
                    qp=q+dq
                    if np.any(qp<self.robot.lower) or np.any(qp>self.robot.upper):continue
                    Tbc=self.robot.fk(qp)@self.T_EC
                    uv=self.tracker.predict_pixel(t+h,.1,Tbc,K,D)
                    row['pixels'][str(round(h*1000))]=uv.tolist()
                    if np.max(abs(uv))<10000:
                        pixel=tuple(np.rint(uv).astype(int));cv2.drawMarker(image,pixel,color,cv2.MARKER_CROSS,20,2)
                        cv2.putText(image,f'{h*1000:.0f}ms',(pixel[0]+10,pixel[1]+15),0,.5,color,1)
                cv2.putText(image,f'Tongji EKF | {row["speed_m_s"]:.3f} m/s | camera CV forecast',(12,30),0,.6,(255,255,0),2)
            except (ValueError,RuntimeError) as e:row['reason']=str(e)
            self.preview.publish(image)
            message=String();message.data=json.dumps(row);self.diagnostics.publish(message)
    rclpy.init();node=Vision()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        node.preview.close();node.tracker.close();node.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
