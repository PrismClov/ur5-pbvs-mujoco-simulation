from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import unittest
import sys
import json
import importlib.util
from pathlib import Path
import numpy as np
from nova5_tongji.pixel_probe import PixelProbe
from nova5_tongji.response_probe import ProbeComplete

@dataclass(frozen=True)
class Config:
    gain:float=.7
    standoff_m:float=.5

class PixelTests(unittest.TestCase):
    def test_analysis_does_not_invent_200ms_after_early_abort(self):
        path=Path(__file__).resolve().parents[1]/'scripts/analyze_pixel_probe.py'
        spec=importlib.util.spec_from_file_location('pixel_analysis_test',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rows=[dict(t=10+t,return_time=10,phase='recover',gain=1.,result=None,
                       center_error_px=[30.,0.],qdot_raw=[0.]*6) for t in (0,.05,.1)]
            (root/'pixel_probe.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            (root/'runtime.log').write_text('RuntimeError: experimental stop\n')
            result=module.analyze(root)
            self.assertIsNone(result['error_at_200ms_px'])
            self.assertEqual(result['result'],'aborted')

    def test_measured_offset_precedes_recovery_and_dwell(self):
        with tempfile.TemporaryDirectory() as path:
            servo=SimpleNamespace(config=Config(),end_to_camera=np.eye(4),
                kinematics=SimpleNamespace(forward=lambda q:np.eye(4)),
                desired_camera_to_marker=lambda:np.eye(4),
                command=lambda q,m:SimpleNamespace(q_velocity_raw=np.zeros(6),rotation_error_deg=0.,distance_error_m=0.))
            probe=PixelProbe(servo,path,30)
            measurement=SimpleNamespace(source_monotonic=0,intrinsics=SimpleNamespace(cx=640,cy=360,fx=1000))
            marker=np.eye(4);marker[2,3]=.5
            def tick(t,x):
                obs=SimpleNamespace(marker_pose_camera=marker,features=np.tile([640+x,360],(4,1)))
                with patch('nova5_tongji.pixel_probe.time.monotonic',return_value=t):
                    probe.observe(np.zeros(6),obs,measurement,True)
            try:
                tick(0,0);tick(4.1,0)
                self.assertEqual(probe.phase,'offset');self.assertAlmostEqual(probe.offset_m,.015)
                probe(np.zeros(6),marker);self.assertEqual(servo.config.gain,.5)
                tick(5,28);self.assertEqual(probe.phase,'offset')
                tick(5.4,28);self.assertEqual(probe.phase,'recover');self.assertEqual(probe.offset_m,0)
                probe(np.zeros(6),marker);self.assertEqual(servo.config.gain,.7)
                tick(5.6,2);tick(5.95,2);self.assertEqual(probe.phase,'hold')
                with self.assertRaises(ProbeComplete):tick(7,2)
            finally:probe.close()

    def test_offset_magnitude_is_bounded(self):
        for pixels in (0,9,401,1000):
            with self.assertRaises(ValueError):PixelProbe(None,None,pixels)

    def test_larger_offsets_preserve_speed_budget_with_roundoff(self):
        with tempfile.TemporaryDirectory() as path:
            rng=np.random.default_rng(4)
            servo=SimpleNamespace(config=Config(),desired_camera_to_marker=lambda:np.eye(4),command=None)
            servo.command=lambda q,m:SimpleNamespace(q_velocity_raw=rng.normal(size=6))
            probe=PixelProbe(servo,path,100)
            try:
                for _ in range(1000):
                    command=probe(np.zeros(6),np.eye(4))
                    self.assertLessEqual(np.linalg.norm(command.q_velocity_raw),.20+1e-12)
            finally:probe.close()

    def test_preflight_includes_centering_from_existing_offset(self):
        def fk(q):
            T=np.eye(4);T[:3,3]=np.asarray(q)[:3]+[0,0,.24];return T
        kin=SimpleNamespace(forward=fk,lower_limits=np.full(6,-2.),upper_limits=np.full(6,2.),
                            link_transforms=lambda q:[fk(q)])
        with tempfile.TemporaryDirectory() as path:
            servo=SimpleNamespace(config=Config(),end_to_camera=np.eye(4),kinematics=kin,
                desired_camera_to_marker=lambda:np.eye(4),command=lambda q,m:None,
                end_jacobian=lambda q:np.eye(6))
            probe=PixelProbe(servo,path,100)
            fake=SimpleNamespace(rotation_vector=lambda R:np.zeros(3))
            try:
                with patch.dict(sys.modules,{'visual_servo.pose_control':fake}):
                    probe.preflight(np.zeros(6),SimpleNamespace(fx=1000),80)
                    self.assertAlmostEqual(probe.plan['travel_min'],-.04)
                    self.assertAlmostEqual(probe.plan['travel_max'],.01)
                    kin.lower_limits=np.zeros(6)
                    with self.assertRaisesRegex(RuntimeError,'preflight failed'):
                        probe.preflight(np.zeros(6),SimpleNamespace(fx=1000),80)
            finally:probe.close()

if __name__=='__main__':unittest.main()
