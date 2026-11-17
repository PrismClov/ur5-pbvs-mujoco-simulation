"""Python binding to the ORIGINAL compiled Tongji EKF, not a replacement filter.

Adapter contract matches the repository's Nova5 PBVS Controller. State models
translation and Base yaw; roll/pitch are held at their last accepted observation.
"""
import ctypes as C
import os
from pathlib import Path
import numpy as np
from nova5_visual_servo.core import pose, exp_so3, Fault


def library_path():
    if os.environ.get('TONGJI_EKF_LIBRARY'):
        return Path(os.environ['TONGJI_EKF_LIBRARY']).resolve()
    root=Path(__file__).resolve().parents[2]
    for name in ('libtongji_conveyor.so','libtongji_conveyor.dll','tongji_conveyor.dll','libtongji_conveyor.dylib'):
        candidate=root/'build'/'lib'/name
        if candidate.exists(): return candidate
    raise FileNotFoundError('Build nova5_tongji with CMake first, or set TONGJI_EKF_LIBRARY')


class TongjiTargetTracker:
    def __init__(self, max_gap=.15, max_speed=1., max_omega=2., position_std_m=.006,
                 orientation_std_rad=.03, linear_accel_psd_m2_s3=.09,
                 angular_accel_psd_rad2_s3=.2):
        self.lib=C.CDLL(str(library_path()))
        ptr=C.POINTER(C.c_double)
        self.lib.tj_create.argtypes=[C.c_double]*7; self.lib.tj_create.restype=C.c_void_p
        self.lib.tj_destroy.argtypes=[C.c_void_p]; self.lib.tj_destroy.restype=None
        self.lib.tj_update.argtypes=[C.c_void_p,C.c_double,ptr,ptr]; self.lib.tj_update.restype=C.c_int
        self.lib.tj_predict.argtypes=[C.c_void_p,C.c_double,ptr,ptr,ptr,C.POINTER(C.c_int)]
        self.lib.tj_predict.restype=C.c_int
        self.handle=self.lib.tj_create(linear_accel_psd_m2_s3,angular_accel_psd_rad2_s3,
                     position_std_m,orientation_std_rad,max_gap,max_speed,max_omega)
        if not self.handle: raise ValueError('invalid EKF configuration')
        self.t=None; self.last_sequence=None; self.ready=False
        self.v=np.zeros(3); self.w=np.zeros(3); self.R=None
        self.yaw=None; self.nis=0.; self.rejections=0; self.last_status='uninitialized'
        self.orientation_std=orientation_std_rad
        self.max_gap=max_gap

    @staticmethod
    def pointer(a): return a.ctypes.data_as(C.POINTER(C.c_double))

    def close(self):
        if getattr(self,'handle',None):
            self.lib.tj_destroy(self.handle);self.handle=None

    def __del__(self): self.close()

    def update(self, sequence, timestamp, T_BE, T_EC, T_CQ):
        if self.last_sequence is not None and sequence <= self.last_sequence:
            raise ValueError('duplicate/out-of-order image sequence')
        if not np.isfinite(timestamp) or (self.t is not None and timestamp <= self.t):
            raise ValueError('image timestamp must increase')
        camera=np.ascontiguousarray(pose(T_BE)@pose(T_EC),dtype=np.float64)
        marker=np.ascontiguousarray(pose(T_CQ),dtype=np.float64)
        if not self.handle: raise RuntimeError('tracker closed')
        code=self.lib.tj_update(self.handle,timestamp,self.pointer(camera),self.pointer(marker))
        self.last_sequence=sequence
        if code <= 0:
            self.rejections+=1; self.last_status='rejected' if code==0 else 'invalid'
            if self.t is not None: self.nis=self._state(self.t)[2]
            return False
        self.t=float(timestamp)
        world=camera@marker
        self.R=world[:3,:3].copy(); self.yaw=float(np.arctan2(self.R[1,0],self.R[0,0]))
        state,P,nis,count=self._state(self.t)
        self.v=state[3:6].copy(); self.w=np.array([0.,0.,state[7]])
        self.nis=nis; self.ready=count>=8
        self.last_status='initialized' if code==1 else 'updated'
        return True

    def _state(self,t):
        if not self.handle: raise RuntimeError('tracker closed')
        x=np.empty(8,np.float64);P=np.empty((8,8),np.float64)
        nis=C.c_double(); count=C.c_int()
        code=self.lib.tj_predict(self.handle,t,self.pointer(x),self.pointer(P),C.byref(nis),C.byref(count))
        if code: raise Fault('Tongji prediction unavailable / invalid time')
        return x,P,nis.value,count.value

    def predict_state(self,t,max_horizon):
        if (self.t is None or not np.isfinite([t,max_horizon]).all() or max_horizon<0
                or t-self.t < -1e-9 or t-self.t > min(max_horizon,.20)+1e-9):
            raise Fault('visual prediction expired')
        x,P,nis,count=self._state(max(t,self.t))
        position_sigma=float(np.sqrt(max(0,np.linalg.eigvalsh(P[:3,:3]).max())))
        angle_sigma=float(max(self.orientation_std,np.sqrt(max(0,P[6,6]))))
        T=np.eye(4);T[:3,3]=x[:3]
        # Only yaw motion is estimated. Never invent roll/pitch angular velocity.
        T[:3,:3]=exp_so3(np.array([0.,0.,x[6]-self.yaw]))@self.R
        usable=count>=8 and position_sigma<.05 and angle_sigma<.35
        return dict(pose=T,position=x[:3],velocity=x[3:6],covariance=P,
                    position_sigma_max_m=position_sigma,orientation_sigma_max_rad=angle_sigma,
                    usable=usable,prior_nis=nis,accepted_count=count,model='single_marker_translation_base_yaw')

    def predict(self,t,max_horizon):
        result=self.predict_state(t,max_horizon)
        if not result['usable']: raise Fault('target estimate not yet usable')
        return result['pose']

    def predict_pixel(self,t,max_horizon,T_BC,K,distortion=None):
        import cv2
        state=self.predict_state(t,max_horizon)
        if not state['usable']: raise Fault('target estimate not yet usable')
        point=np.linalg.inv(pose(T_BC))@np.r_[state['position'],1.]
        if point[2]<=.03: raise Fault('predicted target behind camera')
        pixels,_=cv2.projectPoints(point[:3].reshape(1,3),np.zeros(3),np.zeros(3),
                                  np.asarray(K,float),None if distortion is None else np.asarray(distortion,float))
        return pixels.reshape(2)
