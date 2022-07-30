from setuptools import setup

setup(
    name='src',
    version='0.0.1',
    author='Arnab Mitra',
    description='Face Monitering System',
    packages=['src'],
    install_requires=['torch','torchvision','torchaudio','opencv-python','matplotlib','PyYAML','numpy','scipy']
)