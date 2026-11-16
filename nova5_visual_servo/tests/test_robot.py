import unittest
from pathlib import Path
import numpy as np
from nova5_visual_servo.robot import URDFRobot
from nova5_visual_servo.core import log_so3
from nova5_visual_servo.history import JointHistory


class Tests(unittest.TestCase):
    def test_official_urdf_jacobian(self):
        robot = URDFRobot(Path(__file__).parents[1]/'models/nova5_robot.urdf')
        for q in (np.zeros(6), np.array([.2, -.5, .7, .3, -.2, .1])):
            T, J = robot.evaluate(q)
            numeric = np.zeros((6, 6))
            for i in range(6):
                d = np.zeros(6); d[i] = 1e-6
                Tp, Tm = robot.fk(q+d), robot.fk(q-d)
                numeric[:, i] = np.r_[(Tp[:3, 3]-Tm[:3, 3])/2e-6,
                    log_so3(Tp[:3, :3]@Tm[:3, :3].T)/2e-6]
            np.testing.assert_allclose(J, numeric, atol=1e-7)

    def test_interpolation_and_gap(self):
        h = JointHistory(.05)
        h.add(1., np.zeros(6)); h.add(1.02, np.ones(6))
        np.testing.assert_allclose(h.at(1.01), .5)
        with self.assertRaises(ValueError):
            h.at(1.03)
        with self.assertRaises(ValueError):
            h.add(.9, np.zeros(6))


if __name__ == '__main__':
    unittest.main()
