from setuptools import setup, find_packages
from glob import glob
setup(name='nova5_visual_servo', version='0.1.0', packages=find_packages(),
 data_files=[('share/ament_index/resource_index/packages',['resource/nova5_visual_servo']),
 ('share/nova5_visual_servo',['package.xml']),
 ('share/nova5_visual_servo/config',glob('config/*.json')),
 ('share/nova5_visual_servo/models',glob('models/*.urdf')),
 ('share/nova5_visual_servo/launch',glob('launch/*.py'))],
 install_requires=['numpy'], entry_points={'console_scripts':[
 'servo = nova5_visual_servo.interface:main', 'vision = nova5_visual_servo.vision:main',
 'simulate = nova5_visual_servo.simulate:main']})
