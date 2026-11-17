"""Native Tongji target estimator on the verified V4L2/TCP execution runtime.

Keep legacy runtime source and its safety/limiter/ServoJ path unchanged.
TONGJI_LEGACY_PROJECT selects that installed runtime; shadow is the default.
"""
import hashlib,json,os,sys
from pathlib import Path
import numpy as np
from .tracker import TongjiTargetTracker

def main():
    root=Path(os.environ['TONGJI_LEGACY_PROJECT']).resolve()
    source=root/'src/visual_servo/realtime_tracker.py'
    if hashlib.sha256(source.read_bytes()).hexdigest()!='3038fb87525ee3ed9ba53bfc13741ea9d25e2aa395aebf515290a0abbb03417a':
        raise RuntimeError('Execution runtime differs from verified version')
    mode=os.environ.get('TONGJI_TCP_MODE','shadow')
    if mode not in ('shadow','control'):raise ValueError('invalid mode')
    sys.path.insert(0,str(root/'src'))
    from visual_servo import realtime_tracker as runtime
    class Tracker(runtime.RealtimeTracker):
        def __init__(self,args):
            if mode=='shadow' and (args.execute or args.auto_arm):raise ValueError('shadow cannot execute')
            if args.experiment_dir is None:raise ValueError('experiment archive required')
            super().__init__(args)
            if not self.hand_eye_calibrated:raise ValueError('hand-eye calibration required')
            self.native=TongjiTargetTracker()
            self.native_log=(Path(args.experiment_dir)/'tongji.jsonl').open('x')
            self.scaling=None
            if os.environ.get('TONGJI_GLOBAL_SPEED_PERCENT'):
                if mode!='control':raise ValueError('scaling override requires control mode')
                from .scaling import GlobalScalingOverride
                self.scaling=GlobalScalingOverride(self.robot,int(os.environ['TONGJI_GLOBAL_SPEED_PERCENT']),args.experiment_dir)
            self.probe=None
            if os.environ.get('TONGJI_PIXEL_PROBE'):
                if mode!='control' or os.environ.get('TONGJI_RESPONSE_PROBE')=='1':
                    raise ValueError('pixel probe requires exclusive control mode')
                from .pixel_probe import PixelProbe
                self.probe=PixelProbe(self.pose,args.experiment_dir,float(os.environ['TONGJI_PIXEL_PROBE']))
                self.pose.command=self.probe
            if os.environ.get('TONGJI_RESPONSE_PROBE')=='1':
                if mode!='control':raise ValueError('response probe requires control mode')
                from .response_probe import ResponseProbe
                self.probe=ResponseProbe(self.pose,args.experiment_dir)
                self.pose.command=self.probe
            print('Estimator: original Tongji C++ EKF / TCP '+mode,flush=True)

        def _target_pose_for_control(self,q,observation,measurement,update_filter):
            if self.probe is not None and hasattr(self.probe,'observe'):
                self.probe.observe(q,observation,measurement,update_filter)
            baseline=super()._target_pose_for_control(q,observation,measurement,update_filter)
            result=(self.pose.desired_camera_to_marker(),np.zeros(3),float('nan'),'prediction_unavailable')
            row=dict(sequence=measurement.sequence,t=measurement.source_monotonic,mode=mode,
                     aligned=update_filter,time_quality='arrival_proxy_not_capture',usable=False)
            if update_filter:
                Tbe=self.kinematics.forward(q);Tbc=Tbe@self.end_to_camera
                accepted=self.native.update(measurement.sequence,measurement.source_monotonic,
                    Tbe,self.end_to_camera,observation.marker_pose_camera)
                row.update(accepted=accepted,status=self.native.last_status,prior_nis=self.native.nis)
                if accepted:
                    estimate=self.native.predict_state(measurement.source_monotonic+self.target_prediction_horizon,.15)
                    row.update(usable=estimate['usable'],speed=float(np.linalg.norm(estimate['velocity'])),
                        position_sigma=estimate['position_sigma_max_m'])
                    if estimate['usable']:
                        candidate=np.linalg.inv(Tbc)@estimate['pose']
                        # Retain proven rotation filtering; only translation estimation
                        # and linear velocity FF switch to Tongji in this experiment.
                        candidate[:3,:3]=baseline[0][:3,:3]
                        delta=float(np.linalg.norm(candidate[:3,3]-baseline[0][:3,3]))
                        row['baseline_delta_m']=delta
                        if delta>.03:raise RuntimeError('native/baseline disagreement >30 mm')
                        sigma=float(np.sqrt(np.trace(estimate['covariance'][3:6,3:6])))
                        row['velocity_sigma_m_s']=sigma
                        result=(candidate,estimate['velocity'],sigma,'updated')
            self.native_log.write(json.dumps(row,allow_nan=False)+'\n');self.native_log.flush()
            return baseline if mode=='shadow' else result

        def run(self):
            from .response_probe import ProbeComplete
            try:return super().run()
            except ProbeComplete as exc:print(str(exc),flush=True)
            finally:
                if self.scaling is not None:self.scaling.close()
                if self.probe is not None:self.probe.close()
                self.native_log.close();self.native.close()
    runtime.RealtimeTracker=Tracker
    runtime.main()

if __name__=='__main__':main()
