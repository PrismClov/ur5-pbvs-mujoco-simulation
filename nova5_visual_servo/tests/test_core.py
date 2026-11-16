import unittest
import numpy as np
from nova5_visual_servo.core import *


def translation(p):
    T = np.eye(4)
    T[:3, 3] = p
    return T


def fixture(**changes):
    # Synthetic Cartesian test plant, NOT a Nova5 kinematic model or device limits.
    args = dict(v_max=np.ones(6), a_max=np.ones(6)*2,
                q_min=-np.ones(6)*10, q_max=np.ones(6)*10,
                follow_max=np.ones(6), kp=np.ones(6)*2,
                task_scale=np.ones(6), damping=.01, max_visual_age=.15,
                max_feedback_age=.1, max_prediction=.2, dt_min=.005, dt_max=.04)
    args.update(changes)
    tr = TargetTracker(.15, 3., 3.)
    tr.seq = -2
    tr.update(-1, -.02, np.eye(4), np.eye(4), np.eye(4))
    tr.update(0, 0., np.eye(4), np.eye(4), np.eye(4))
    def fk(q):
        T = translation(q[:3])
        T[:3, :3] = exp_so3(q[3:])
        return T
    ctrl = Controller(Config(**args), tr, np.eye(4), np.eye(4), fk,
                      lambda q: np.eye(6), lambda q: True,
                      np.zeros(6), np.zeros(6), 0.)
    return tr, ctrl


class Tests(unittest.TestCase):
    def test_rotation_roundtrip(self):
        for v in ([0., 0., 0.], [.1, -.3, .2], [np.pi, 0., 0.]):
            R = exp_so3(np.array(v))
            np.testing.assert_allclose(exp_so3(log_so3(R)), R, atol=1e-7)

    def test_camera_motion_not_target_motion(self):
        tr, _ = fixture()
        tr.update(1, .03, translation([.1, 0, 0]), np.eye(4), translation([-.1, 0, 0]))
        np.testing.assert_allclose(tr.v, 0, atol=1e-12)

    def test_duplicate_frame_does_not_update(self):
        tr, _ = fixture()
        self.assertFalse(tr.update(0, .03, np.eye(4), np.eye(4), translation([1, 0, 0])))
        self.assertEqual(tr.t, 0.)

    def test_prediction_and_feedforward(self):
        tr, ctrl = fixture(execution_lead=.02)
        tr.update(1, .02, np.eye(4), np.eye(4), translation([.01, 0, 0]))
        out = ctrl.tick(.02, np.zeros(6), np.zeros(6), .02)
        self.assertGreater(out['ff'][0], 0.)
        self.assertAlmostEqual(out['error'][0], tr.T[0, 3]+tr.v[0]*.02)
        self.assertAlmostEqual(out['execution_time'], .04)

    def test_rotating_offset_feedforward(self):
        tr, ctrl = fixture()
        ctrl.T_QE_des = translation([1, 0, 0])
        tr.w = np.array([0., 0., 1.])
        tr.ready = True
        out = ctrl.tick(.02, np.zeros(6), np.zeros(6), .02)
        np.testing.assert_allclose(out['ff'][:3], [-np.sin(.02), np.cos(.02), 0], atol=1e-12)

    def test_velocity_acceleration_and_integral(self):
        tr, ctrl = fixture()
        tr.T[0, 3] = 1.
        out = ctrl.tick(.02, np.zeros(6), np.zeros(6), .02)
        self.assertLessEqual(abs(out['v_next'][0]), 2*.02+1e-12)
        self.assertAlmostEqual(out['q_plan'][0], .5*out['v_next'][0]*.02)
        self.assertTrue(out['velocity_limited'])

    def test_stale_feedback_latches(self):
        _, ctrl = fixture()
        with self.assertRaises(Fault):
            ctrl.tick(.02, np.zeros(6), np.zeros(6), -.2)
        with self.assertRaises(Fault):
            ctrl.tick(.02, np.zeros(6), np.zeros(6), .02)

    def test_outlier_does_not_refresh_age(self):
        tr, _ = fixture()
        self.assertFalse(tr.update(1, .02, np.eye(4), np.eye(4), translation([1, 0, 0])))
        self.assertEqual(tr.t, 0.)

    def test_joint_limit_fault(self):
        tr, ctrl = fixture(q_max=np.ones(6)*.0001)
        tr.T[0, 3] = 1.
        with self.assertRaises(Fault):
            ctrl.tick(.02, np.zeros(6), np.zeros(6), .02)

    def test_tracking_feedforward_reduces_error(self):
        def run(ff):
            tr, ctrl = fixture(feedforward=ff)
            q, qd = np.zeros(6), np.zeros(6)
            errors = []
            for k in range(1, 251):
                t = k*.02
                if k % 2 == 0:
                    tr.update(k, t, translation(q[:3]), np.eye(4),
                              translation([.05*t-q[0], -q[1], -q[2]]))
                out = ctrl.tick(t, q, qd, t)
                qd = (out['q_plan']-q)/.02
                q = out['q_plan']
                errors.append(abs(.05*t-q[0]))
            return np.mean(errors[-50:])
        self.assertLess(run(True), .3*run(False))


if __name__ == '__main__':
    unittest.main(verbosity=2)

