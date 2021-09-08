from distutils.core import setup
from setuptools import find_packages

setup(
    name='graspnetAPI',
    version='0.1',
    description='graspnet APT',
    author='Hao-shu Fang, Chenxi Wang, Minghao Gou',
    author_email='fhaoshu@gmail.com',
    url='graspnet.net',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'transforms3d==0.3.1',
        'open3d==0.8.0',
        'trimesh==3.7.14',
        'tqdm==4.48.2',
        'Pillow==8.3.2',
        'opencv-python'
    ]
)