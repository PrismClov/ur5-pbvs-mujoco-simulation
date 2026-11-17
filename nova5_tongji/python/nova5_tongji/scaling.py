"""Set only global scaling after legacy startup, then verify fresh feedback."""
import json
import time
from pathlib import Path


class GlobalScalingOverride:
    def __init__(self,robot,percent,path):
        if not 1<=percent<=100:raise ValueError('global scaling must be 1..100')
        self.robot=robot;self.percent=percent;self.original=robot.connect
        self.latest=None;self.last_log=0.;self.log=(Path(path)/'speed_scaling.jsonl').open('x')
        robot.connect=self.connect

    def connect(self):
        self.original()
        if self.robot.robot_mode()!=5:raise RuntimeError('scaling change requires idle enabled robot')
        reader=self.robot.feedback
        if reader is None:raise RuntimeError('speed scaling feedback unavailable')
        original_read=reader.feedBackData
        def read():
            packet=original_read()
            if packet is not None:
                self.latest=dict(t=time.monotonic(),speed=float(packet['SpeedScaling'][0]),
                    velocity_ratio=float(packet['VelocityRatio'][0]),acceleration_ratio=float(packet['AccelerationRatio'][0]))
                if self.latest['t']-self.last_log>=.5:
                    self.log.write(json.dumps(self.latest)+'\n');self.log.flush();self.last_log=self.latest['t']
            return packet
        reader.feedBackData=read
        reply=self.robot.dashboard.SpeedFactor(self.percent)
        if not reply or not reply.startswith('0,'):raise RuntimeError('global scaling command not acknowledged')
        deadline=time.monotonic()+2;after=time.monotonic()+.1
        while time.monotonic()<deadline:
            sample=self.latest
            if sample and sample['t']>=after and sample['speed']==self.percent:
                if (sample['velocity_ratio']!=self.robot.speed_factor_percent
                        or sample['acceleration_ratio']!=self.robot.speed_factor_percent):
                    raise RuntimeError('joint ratios unexpectedly changed')
                print('Verified scaling: '+json.dumps(sample),flush=True)
                return
            time.sleep(.01)
        raise RuntimeError('global scaling feedback verification timed out')

    def close(self):
        self.log.close()
