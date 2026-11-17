import json
import threading
import unittest
import urllib.request
from pathlib import Path
import cv2
import numpy as np
from nova5_tongji.tracker import TongjiTargetTracker
from nova5_tongji.vision import detect_pose
from nova5_tongji.preview import Preview
from nova5_visual_servo.core import Config,Controller,Fault,exp_so3
from nova5_visual_servo.robot import URDFRobot

def translation(p):
    T=np.eye(4);T[:3,3]=p;return T

def camera(t):
    T=translation([.06*np.sin(t*2),.01*np.cos(t),0])
    T[:3,:3]=exp_so3(np.array([0,.12*np.sin(t),.1*np.sin(t)]));return T

class Tests(unittest.TestCase):
    def new(self,**kw):
        tr=TongjiTargetTracker(**kw);self.addCleanup(tr.close);return tr

    def test_static_target_moving_camera(self):
        tr=self.new();target=translation([.05,0,.8])
        for i,t in enumerate(np.arange(0,2,.02)):
            T=camera(t);self.assertTrue(tr.update(i,t,T,np.eye(4),np.linalg.inv(T)@target))
        self.assertTrue(tr.ready)
        self.assertLess(np.linalg.norm(tr.v),1e-6)
        np.testing.assert_allclose(tr.predict(tr.t+.1,.15),target,atol=1e-6)

    def test_unknown_velocity_and_future_pixel_with_moving_camera(self):
        tr=self.new();v=np.array([.12,-.03,.01]);rng=np.random.default_rng(36)
        predicted=[];frozen=[]
        for i,t in enumerate(np.arange(0,3,.02)):
            world=translation(np.array([0,0,.8])+v*t);T=camera(t)
            obs=np.linalg.inv(T)@world;obs[:3,3]+=rng.normal(size=3)*.0005
            tr.update(i,t,T,np.eye(4),obs)
            if t>1:
                K=np.array([[500.,0,320],[0,500.,240],[0,0,1]])
                forecast=tr.predict_pixel(t+.1,.15,camera(t+.1),K,np.zeros(5))
                point=np.linalg.inv(camera(t+.1))@np.r_[world[:3,3]+v*.1,1]
                truth=(K@point[:3])[:2]/point[2]
                fixed=np.linalg.inv(camera(t+.1))@np.r_[tr.predict_state(t,.15)['position'],1]
                frozen_pixel=(K@fixed[:3])[:2]/fixed[2]
                predicted.append(np.linalg.norm(forecast-truth))
                frozen.append(np.linalg.norm(frozen_pixel-truth))
        self.assertLess(np.linalg.norm(tr.v-v),.015)
        self.assertLess(np.mean(np.square(predicted)),np.mean(np.square(frozen))*.2)

    def test_prediction_is_copy_not_double_integration(self):
        tr=self.new()
        for i in range(40):tr.update(i,i*.02,np.eye(4),np.eye(4),translation([i*.002,0,.6]))
        a=tr.predict_state(tr.t,.15)
        p=tr.predict(tr.t+.1,.15)
        np.testing.assert_array_equal(p,tr.predict(tr.t+.1,.15))
        np.testing.assert_allclose(p,tr.predict(tr.t+.1,.1))
        np.testing.assert_array_equal(a['covariance'],tr.predict_state(tr.t,.15)['covariance'])
        self.assertGreater(np.trace(tr.predict_state(tr.t+.1,.15)['covariance']),np.trace(a['covariance']))

    def test_user_conveyor_model_30fps_and_dropped_frame(self):
        tr=self.new();velocity=np.array([.1,0.,0.])
        for i in range(60):
            t=i/30
            tr.update(i,t,np.eye(4),np.eye(4),translation(np.array([0,0,.6])+velocity*t))
        current=tr.predict_state(tr.t,.15)
        for dt in (1/30,2/30,.1):
            future=tr.predict_state(tr.t+dt,.15)
            np.testing.assert_allclose(future['position'],current['position']+current['velocity']*dt,atol=1e-12)
            np.testing.assert_allclose(future['velocity'],current['velocity'],atol=1e-12)
        self.assertAlmostEqual((tr.predict_state(tr.t+1/30,.15)['position']-current['position'])[0],.0033333333,delta=.0001)

    def test_outlier_does_not_refresh_time(self):
        tr=self.new()
        for i in range(20):tr.update(i,i*.02,np.eye(4),np.eye(4),translation([0,0,.6]))
        old=tr.t
        self.assertFalse(tr.update(20,old+.02,np.eye(4),np.eye(4),translation([3,3,3])))
        self.assertEqual(tr.t,old)
        with self.assertRaises(Fault):tr.predict(old+.21,.3)

    def test_duplicate_and_reinitialization(self):
        tr=self.new();tr.update(0,1,np.eye(4),np.eye(4),translation([0,0,.6]))
        with self.assertRaises(ValueError):tr.update(0,1.02,np.eye(4),np.eye(4),translation([0,0,.6]))
        tr.update(1,2,np.eye(4),np.eye(4),translation([.3,0,.6]))
        self.assertFalse(tr.ready);np.testing.assert_allclose(tr.v,0)

    def test_yaw_wrap_and_full_measured_orientation(self):
        tr=self.new(max_omega=3)
        for i in range(70):
            target=translation([0,0,.6])
            target[:3,:3]=exp_so3(np.array([0,0,3+i*.01]))@exp_so3(np.array([.1,.1,0]))
            tr.update(i,i*.02,np.eye(4),np.eye(4),target)
        self.assertLess(abs(tr.w[2]-.5),.03)
        R=tr.predict(tr.t+.05,.15)[:3,:3]
        expected=exp_so3(np.array([0,0,.025]))@target[:3,:3]
        np.testing.assert_allclose(R,expected,atol=.01)

    def test_native_config_and_bad_projection(self):
        with self.assertRaises(ValueError):self.new(position_std_m=-1)
        tr=self.new();self.assertFalse(tr.update(0,0,np.eye(4),np.eye(4),translation([0,0,-.6])))

    def test_existing_nova5_controller_uses_native_estimate(self):
        tr=self.new()
        for i in range(40):tr.update(i,i*.02,np.eye(4),np.eye(4),translation([i*.0004,0,.6]))
        cfg=Config(v_max=np.ones(6)*.1,a_max=np.ones(6)*.3,q_min=-np.ones(6)*3,q_max=np.ones(6)*3,
            follow_max=np.ones(6)*.2,kp=np.ones(6),task_scale=np.ones(6),damping=.03,
            max_visual_age=.12,max_feedback_age=.05,max_prediction=.15,dt_min=.005,dt_max=.04)
        def fk(q):
            T=translation(q[:3]);T[:3,:3]=exp_so3(q[3:]);return T
        ctrl=Controller(cfg,tr,translation([0,0,-.6]),np.eye(4),fk,lambda q:np.eye(6),lambda q:True,
                        np.zeros(6),np.zeros(6),tr.t)
        out=ctrl.tick(tr.t+.02,np.zeros(6),np.zeros(6),tr.t+.02)
        self.assertGreater(out['ff'][0],.005)
        self.assertLessEqual(np.max(abs(out['v_next'])),.3*.02+1e-10)
        with self.assertRaises(Fault):ctrl.tick(tr.t+.5,np.zeros(6),np.zeros(6),tr.t+.5)

    def test_actual_nova5_urdf_camera_jacobian(self):
        path=Path(__file__).resolve().parents[2]/'nova5_visual_servo/models/nova5_robot.urdf'
        robot=URDFRobot(path);q=np.array([.2,-.1,.2,-.1,.15,.2]);T,J=robot.evaluate(q)
        numeric=np.column_stack([(robot.fk(q+np.eye(6)[i]*1e-6)[:3,3]-robot.fk(q-np.eye(6)[i]*1e-6)[:3,3])/2e-6 for i in range(6)])
        np.testing.assert_allclose(J[:3],numeric,atol=1e-7)

    def test_aruco_pose_input_and_wrong_id(self):
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker=(cv2.aruco.generateImageMarker(dictionary,0,200)
                if hasattr(cv2.aruco,'generateImageMarker') else cv2.aruco.drawMarker(dictionary,0,200))
        image=np.ones((480,640),np.uint8)*255;image[140:340,220:420]=marker
        K=np.array([[500.,0,320],[0,500.,240],[0,0,1]])
        found=detect_pose(image,K,np.zeros(5),.1,'aruco',0)
        self.assertIsNotNone(found);self.assertAlmostEqual(found[0][2,3],.25,delta=.01)
        self.assertIsNone(detect_pose(image,K,np.zeros(5),.1,'aruco',1))

    def test_qr_pose_input(self):
        encoded=cv2.imread(str(Path(__file__).parent/'fixtures/nova5-target.png'),cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(encoded)
        code=cv2.resize(encoded,(280,280),interpolation=cv2.INTER_NEAREST)
        image=np.ones((480,640),np.uint8)*255;image[100:380,180:460]=code
        K=np.array([[500.,0,320],[0,500.,240],[0,0,1]])
        self.assertIsNotNone(detect_pose(image,K,np.zeros(5),.1,'qr',text='nova5-target'))
        self.assertIsNone(detect_pose(image,K,np.zeros(5),.1,'qr',text='wrong'))

    def test_http_preview_sends_decodable_jpeg(self):
        server=Preview(port=0);self.addCleanup(server.close)
        server.publish(np.zeros((120,160,3),np.uint8))
        with urllib.request.urlopen('http://127.0.0.1:%d/stream.mjpg'%server.server.server_port,timeout=5) as r:
            self.assertIn(b'--frame',r.readline())
            self.assertIn(b'image/jpeg',r.readline())
            size=int(r.readline().split(b':')[1]);r.readline();raw=r.read(size)
        decoded=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape,(120,160,3))

if __name__=='__main__':unittest.main()
