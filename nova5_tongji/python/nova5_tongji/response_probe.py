"""Opt-in fixed-target, one-degree pitch response experiment (not tracking mode)."""
import json
import time
from pathlib import Path
import numpy as np


class ProbeComplete(RuntimeError):
    """Normal experiment completion; the runtime still executes its stop/finally."""


def reference_pitch(elapsed):
    """Six seconds settling, one 8-second step with an 8-second return."""
    if elapsed < 6 or elapsed >= 22:
        return 0.0
    return np.deg2rad(1.0) if (elapsed-6) % 16 < 8 else 0.0


def pitch_matrix(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])


class ResponseProbe:
    def __init__(self,servo,path):
        self.servo=servo
        self.original=servo.command
        self.log=(Path(path)/'response_probe.jsonl').open('x')
        self.start=None
        self.initial=None
        self.initial_q=None
        self.last_pitch=None

    def __call__(self,q,marker):
        now=time.monotonic()
        current=self.servo.kinematics.forward(q)
        if self.start is None:
            self.start=now
            self.initial=current.copy()
            self.initial_q=np.asarray(q).copy()
        elapsed=now-self.start
        angle=reference_pitch(elapsed)
        if elapsed>26:
            raise ProbeComplete('response probe complete; stopped')
        displacement=np.linalg.norm(current[:3,3]-self.initial[:3,3])
        if displacement>.03 or np.max(np.abs(np.asarray(q)-self.initial_q))>.08:
            raise RuntimeError('response probe motion envelope exceeded')
        self.servo.DESIRED_CAMERA_TO_MARKER_ROTATION=pitch_matrix(angle)
        command=self.original(q,marker)
        if not np.isfinite(command.q_velocity_raw).all() or np.linalg.norm(command.q_velocity_raw)>.15:
            raise RuntimeError('response probe command envelope exceeded')
        row=dict(t=now,elapsed=elapsed,reference_pitch_rad=angle,
                 q=np.asarray(q).tolist(),current_pose=current.tolist(),
                 desired_pose=command.desired_end_pose.tolist(),
                 marker_pose=np.asarray(marker).tolist(),
                 qdot_raw=command.q_velocity_raw.tolist(),
                 rotation_error_deg=command.rotation_error_deg,
                 position_error_m=command.distance_error_m,
                 gain=self.servo.config.gain)
        self.log.write(json.dumps(row,allow_nan=False)+'\n')
        self.log.flush()
        if angle!=self.last_pitch:
            print('Response probe: t=%.3fs pitch=%.1fdeg gain=%.2f' %
                  (elapsed,np.rad2deg(angle),self.servo.config.gain),flush=True)
            self.last_pitch=angle
        return command

    def close(self):
        self.servo.DESIRED_CAMERA_TO_MARKER_ROTATION=np.eye(3)
        self.servo.command=self.original
        self.log.close()
