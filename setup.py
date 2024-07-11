from setuptools import setup, find_packages

setup(
    name='HydraNet',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'tensorboardX==2.6.2.2',    
        'pyyaml==6.0.1',
        'opencv-python==4.10.0.84',
        'tqdm==4.66.4',
        'tensorboard==2.17.0',
        'torch==2.3.1',
        'torchvision==0.18.1',
        'torchaudio==2.3.1'
    ],
    author='Vinay Purushotham',
    author_email='vp32@illinois.edu',
    description='A package for HydraNet',
    url='https://github.com/vinaypnaidu/Tesla-Vision-Stack'
)