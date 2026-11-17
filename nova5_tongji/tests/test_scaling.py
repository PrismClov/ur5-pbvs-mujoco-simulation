import tempfile
import threading
from types import SimpleNamespace
import unittest
from nova5_tongji.scaling import GlobalScalingOverride


class ScalingTests(unittest.TestCase):
    def exercise(self,reject=False):
        with tempfile.TemporaryDirectory() as path:
            stop=threading.Event();calls=[];threads=[]
            robot=SimpleNamespace(speed_factor_percent=20,speed=20,robot_mode=lambda:5)
            def set_speed(value):
                calls.append(('SpeedFactor',value))
                if reject:return '-1,{},SpeedFactor();'
                robot.speed=value;return '0,{},SpeedFactor();'
            robot.dashboard=SimpleNamespace(SpeedFactor=set_speed)
            robot.feedback=SimpleNamespace(feedBackData=lambda:dict(
                SpeedScaling=[robot.speed],VelocityRatio=[20],AccelerationRatio=[20]))
            def start():
                def loop():
                    while not stop.wait(.005):robot.feedback.feedBackData()
                thread=threading.Thread(target=loop);thread.start();threads.append(thread)
            robot.connect=start
            override=GlobalScalingOverride(robot,100,path)
            try:
                if reject:
                    with self.assertRaisesRegex(RuntimeError,'not acknowledged'):robot.connect()
                else:
                    robot.connect()
                    self.assertEqual(override.latest['speed'],100)
                    self.assertEqual(override.latest['velocity_ratio'],20)
                    self.assertEqual(override.latest['acceleration_ratio'],20)
                self.assertEqual(calls,[('SpeedFactor',100)])
            finally:
                stop.set()
                for thread in threads:thread.join()
                override.close()

    def test_global_only_verified_from_fresh_feedback(self):self.exercise()
    def test_rejected_command_aborts(self):self.exercise(reject=True)

if __name__=='__main__':unittest.main()
