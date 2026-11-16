import unittest
import numpy as np
from nova5_visual_servo.core import TargetTracker, exp_so3, log_so3, Fault
from nova5_visual_servo.target_pose_ekf import transition, process_noise, left_jacobian


class ESKFTests(unittest.TestCase):
    def tracker(self):
        tr = TargetTracker(.2, 3., 3.)
        tr.update(0, 0., np.eye(4), np.eye(4), np.eye(4))
        tr.update(1, .03, np.eye(4), np.eye(4), np.eye(4))
        return tr

    def test_transition_numerical(self):
        w = np.array([.3, -.7, .5]); h = .08
        R = exp_so3(np.array([.2, .1, -.3]))
        base = exp_so3(w*h)@R
        F = transition(w, h)
        for i in range(6):
            d = np.zeros(6); d[i] = 1e-7
            perturbed = exp_so3((w+d[3:])*h)@exp_so3(d[:3])@R
            numeric = log_so3(perturbed@base.T)/1e-7
            np.testing.assert_allclose(numeric, F[6:9, 6+i], atol=1e-7)

    def test_reset_numerical(self):
        a = np.array([.2, -.1, .3])
        J = left_jacobian(a)
        for i in range(3):
            d = np.zeros(3); d[i] = 1e-7
            n = log_so3(exp_so3(a+d)@exp_so3(-a))/1e-7
            np.testing.assert_allclose(n, J[:, i], atol=1e-7)

    def test_noise_composition(self):
        w = np.array([.3, .5, -.7])
        Q = lambda h: process_noise(w, h, .02, .2)
        F = transition(w, .07)
        np.testing.assert_allclose(Q(.1), F@Q(.03)@F.T+Q(.07), atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(Q(.1)).min(), -1e-12)

    def test_prediction_is_read_only(self):
        tr = self.tracker(); P, T = tr.P.copy(), tr.T.copy()
        a = tr.predict_state(.08, .2); b = tr.predict_state(.08, .2)
        np.testing.assert_array_equal(a['T'], b['T'])
        np.testing.assert_array_equal(tr.P, P)
        np.testing.assert_array_equal(tr.T, T)
        self.assertGreater(np.trace(a['covariance']), np.trace(P))

    def test_rejects_then_recovers(self):
        tr = self.tracker(); old = tr.t
        Z = np.eye(4); Z[0, 3] = 10.
        for k in range(2, 5):
            self.assertFalse(tr.update(k, k*.03, np.eye(4), np.eye(4), Z))
        self.assertEqual(tr.t, old)
        self.assertFalse(tr.predict_state(.12, .2)['usable'])
        self.assertTrue(tr.update(5, .15, np.eye(4), np.eye(4), np.eye(4)))
        self.assertTrue(tr.predict_state(.15, .2)['usable'])

    def test_order_gap_uncertainty(self):
        tr = self.tracker()
        self.assertFalse(tr.update(9, .02, np.eye(4), np.eye(4), np.eye(4)))
        with self.assertRaises(Fault):
            tr.update(10, .4, np.eye(4), np.eye(4), np.eye(4))
        with self.assertRaises(Fault):
            tr.predict(.5, .1)
        tr.max_position_sigma = 1e-6
        self.assertFalse(tr.predict_state(.04, .1)['usable'])

    def test_velocity_and_rotation_cross_pi(self):
        tr = TargetTracker(.2, 3., 3.)
        for k in range(150):
            t = k*.03
            T = np.eye(4); T[:3, :3] = exp_so3(np.array([0., 0., 3.+.2*t]))
            T[0, 3] = .1*t
            self.assertTrue(tr.update(k, t, np.eye(4), np.eye(4), T))
        np.testing.assert_allclose(tr.v, [.1, 0, 0], atol=.002)
        np.testing.assert_allclose(tr.w, [0, 0, .2], atol=.002)
        self.assertGreaterEqual(np.linalg.eigvalsh(tr.P).min(), -1e-12)
