import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from nova5_tongji.response_probe import reference_pitch,pitch_matrix,ResponseProbe,ProbeComplete

class ProbeTests(unittest.TestCase):
    def test_schedule_bounded_and_returns(self):
        for t in [0,5.999,14,30,46,54,58]:self.assertEqual(reference_pitch(t),0)
        for t in [6,13.999]:self.assertAlmostEqual(reference_pitch(t),np.pi/180)
        R=pitch_matrix(reference_pitch(6))
        np.testing.assert_allclose(R.T@R,np.eye(3),atol=1e-15)

    def test_excessive_command_is_not_returned(self):
        with tempfile.TemporaryDirectory() as path:
            servo=SimpleNamespace(kinematics=SimpleNamespace(forward=lambda q:np.eye(4)),
                command=lambda q,m:SimpleNamespace(q_velocity_raw=np.ones(6)))
            probe=ResponseProbe(servo,path)
            try:
                with self.assertRaisesRegex(RuntimeError,'command envelope'):
                    probe(np.zeros(6),np.eye(4))
            finally:probe.close()

    def test_completion_and_motion_envelope_precede_command(self):
        def forbidden(q,m):raise AssertionError('must not calculate a motion command')
        with tempfile.TemporaryDirectory() as path:
            servo=SimpleNamespace(kinematics=SimpleNamespace(forward=lambda q:np.eye(4)),command=forbidden)
            probe=ResponseProbe(servo,path)
            probe.start=0.;probe.initial=np.eye(4);probe.initial_q=np.zeros(6)
            try:
                with patch('nova5_tongji.response_probe.time.monotonic',return_value=27):
                    with self.assertRaises(ProbeComplete):probe(np.zeros(6),np.eye(4))
                with patch('nova5_tongji.response_probe.time.monotonic',return_value=1):
                    with self.assertRaisesRegex(RuntimeError,'motion envelope'):
                        probe(np.array([.081,0,0,0,0,0]),np.eye(4))
            finally:probe.close()

if __name__=='__main__':unittest.main()
