"""Bounded real-image offset/return experiment for a stationary target."""
from dataclasses import replace
import json
import time
from pathlib import Path
import numpy as np
from .response_probe import ProbeComplete


class PixelProbe:
    def __init__(self,servo,path,pixels=30):
        if not 10<=pixels<=400:raise ValueError('pixel probe range is 10..400 px')
        self.servo=servo;self.pixels=float(pixels)
        self.original=servo.command;self.original_desired=servo.desired_camera_to_marker
        self.config=servo.config;self.offset_m=0.
        self.log=(Path(path)/'pixel_probe.jsonl').open('x')
        self.phase='settle';self.start=None;self.phase_time=None;self.dwell=None
        self.initial=None;self.initial_q=None;self.target_anchor=None
        self.latest=None;self.result=None;self.return_time=None;self.finish_at=None
        self.path=Path(path);self.plan=None
        servo.desired_camera_to_marker=self.desired

    def preflight(self,q,intr,start_error_px=0.):
        """Check IK along the translation; this is not a scene collision check."""
        from visual_servo.pose_control import rotation_vector
        kin=self.servo.kinematics;EC=self.servo.end_to_camera
        q=np.asarray(q).copy();q0=q.copy();C0=kin.forward(q)@EC
        scale=self.config.standoff_m/intr.fx
        lower=min(0.,-start_error_px*scale)
        upper=max(0.,(self.pixels-start_error_px)*scale)
        travel=upper-lower
        history=[q.copy()]
        for d in np.linspace(lower,upper,max(2,int(self.pixels/10)+1)):
            C=C0.copy();C[:3,3]-=C0[:3,0]*d;goal=C@np.linalg.inv(EC)
            for iteration in range(200):
                T=kin.forward(q)
                error=np.r_[goal[:3,3]-T[:3,3],rotation_vector(goal[:3,:3]@T[:3,:3].T)]
                if np.linalg.norm(error)<1e-5:break
                J=self.servo.end_jacobian(q)
                dq=J.T@np.linalg.solve(J@J.T+np.eye(6)*1e-5,error)
                q+=dq*min(1,.02/max(np.linalg.norm(dq),1e-9))
            J=self.servo.end_jacobian(q)
            if (iteration==199 or np.min(q-kin.lower_limits)<.15
                    or np.min(kin.upper_limits-q)<.15
                    or np.linalg.svd(J,compute_uv=False)[-1]<.05
                    or min(T[2,3] for T in kin.link_transforms(q))<.10):
                raise RuntimeError('pixel probe kinematic preflight failed')
            history.append(q.copy())
        H=np.asarray(history)
        self.plan=dict(axis=-C0[:3,0],travel_min=lower,travel_max=upper,
                       lower=H.min(axis=0)-.05,upper=H.max(axis=0)+.05)
        (self.path/'pixel_preflight.json').write_text(json.dumps(dict(
            requested_pixels=self.pixels,translation_m=travel,q_start=q0.tolist(),q_path=H.tolist(),
            tcp_origin=self.servo.kinematics.forward(q0)[:3,3].tolist(),axis=self.plan['axis'].tolist(),
            travel_min=lower,travel_max=upper,cross_track_limit_m=.02,
            q_lower=self.plan['lower'].tolist(),q_upper=self.plan['upper'].tolist(),
            note='Kinematic reachability only; no scene collision model. Workspace cleared by operator.'),indent=2))

    def desired(self):
        T=self.original_desired().copy();T[0,3]+=self.offset_m;return T

    def transition(self,phase,now):
        self.phase=phase;self.phase_time=now;self.dwell=None
        print('Pixel probe: '+phase,flush=True)

    def observe(self,q,observation,measurement,aligned):
        now=time.monotonic();T=self.servo.kinematics.forward(q)
        marker=np.asarray(observation.marker_pose_camera)
        center=np.asarray(observation.features).reshape(-1,2).mean(axis=0)
        intr=measurement.intrinsics
        error=center-np.array([intr.cx,intr.cy])
        target=(T@self.servo.end_to_camera@marker)[:3,3]
        if self.start is None:
            self.start=now;self.phase_time=now;self.initial=T.copy();self.initial_q=np.asarray(q).copy()
            if self.pixels>60:self.preflight(q,intr,float(error[0]))
        if now-self.start>48:raise RuntimeError('pixel probe total timeout')
        if self.plan is None:
            envelope=(np.linalg.norm(T[:3,3]-self.initial[:3,3])<=.04
                      and np.max(np.abs(np.asarray(q)-self.initial_q))<=.12)
        else:
            delta=T[:3,3]-self.initial[:3,3];along=float(delta@self.plan['axis'])
            across=np.linalg.norm(delta-along*self.plan['axis'])
            envelope=(self.plan['travel_min']-.02<=along<=self.plan['travel_max']+.02 and across<=.02
                      and np.all(q>=self.plan['lower']) and np.all(q<=self.plan['upper']))
        if not np.isfinite(error).all() or not envelope:
            fault=dict(t=now,phase='fault',phase_before=self.phase,
                       reason='motion_envelope',q=np.asarray(q).tolist())
            if self.plan is not None:fault.update(along_m=along,cross_m=float(across))
            self.log.write(json.dumps(fault)+'\n');self.log.flush()
            raise RuntimeError('pixel probe motion envelope exceeded')
        if self.target_anchor is not None and np.linalg.norm(target-self.target_anchor)>.02:
            raise RuntimeError('pixel probe target moved or pose became inconsistent')
        if not aligned:raise RuntimeError('pixel probe requires aligned joint feedback')
        if self.phase=='settle':
            if now-self.start>12:raise RuntimeError('pixel probe could not establish centered start')
            if now-self.start>=4 and np.linalg.norm(error)<5:
                self.target_anchor=target.copy()
                self.offset_m=self.pixels*self.config.standoff_m/intr.fx
                if self.offset_m>.23:raise ValueError('pixel offset exceeds calibrated displacement budget')
                self.transition('offset',now)
        elif self.phase=='offset':
            if now-self.phase_time>20:raise RuntimeError('pixel offset was not achieved')
            within=abs(error[0]-self.pixels)<=7 and abs(error[1])<(10 if self.pixels>100 else 5)
            self.dwell=(now if self.dwell is None else self.dwell) if within else None
            if self.dwell is not None and now-self.dwell>=.3:
                self.offset_m=0.;self.return_time=now;self.transition('recover',now)
        elif self.phase=='recover':
            within=np.linalg.norm(error)<=5
            self.dwell=(now if self.dwell is None else self.dwell) if within else None
            if self.dwell is not None and now-self.dwell>=.3:
                self.result='settled';self.finish_at=now+1.;self.transition('hold',now)
            elif now-self.phase_time>=10:
                self.result='not_settled_in_10s';self.finish_at=now;self.transition('hold',now)
        self.latest=dict(t=now,t_measurement=measurement.source_monotonic,
            elapsed=now-self.start,phase=self.phase,center_error_px=error.tolist(),
            requested_offset_px=self.pixels,offset_m=self.offset_m,
            return_time=self.return_time,q=np.asarray(q).tolist(),
            target_position_base=target.tolist(),result=self.result)
        if self.phase=='hold' and now>=self.finish_at:
            self.log.write(json.dumps(self.latest)+'\n');self.log.flush()
            raise ProbeComplete('pixel probe complete: '+self.result)

    def __call__(self,q,marker):
        gain=self.config.gain if self.phase in ('recover','hold') else .5
        self.servo.config=replace(self.config,gain=gain)
        command=self.original(q,marker)
        requested=command.q_velocity_raw.copy()
        # Larger offsets use the same 0.20 rad/s experiment command budget.
        # Keep the request separately so saturation cannot masquerade as low demand.
        norm=float(np.linalg.norm(requested))
        if self.pixels>60 and np.isfinite(norm) and norm>.20:
            command.q_velocity_raw=requested*(.20/norm)
        if (not np.isfinite(command.q_velocity_raw).all()
                or np.linalg.norm(command.q_velocity_raw)>.20+1e-12):
            raise RuntimeError('pixel probe command envelope exceeded')
        if self.latest is not None:
            self.log.write(json.dumps(dict(self.latest,gain=gain,
                qdot_requested=requested.tolist(),experiment_speed_saturated=norm>.20,
                qdot_raw=command.q_velocity_raw.tolist(),
                rotation_error_deg=command.rotation_error_deg,
                position_error_m=command.distance_error_m),allow_nan=False)+'\n')
            self.log.flush()
        return command

    def close(self):
        self.servo.config=self.config;self.servo.command=self.original
        self.servo.desired_camera_to_marker=self.original_desired;self.log.close()
