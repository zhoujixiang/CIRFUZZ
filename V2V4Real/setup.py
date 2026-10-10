# -*- coding: utf-8 -*-


from os.path import dirname, realpath
from setuptools import setup, find_packages
from opencood.version import __version__


def _read_requirements_file():
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f]

setup(
    name='V2V4Real',
    version=__version__,
    packages=find_packages(),
    url='https://github.com/ucla-mobility/V2V4Real',
    license='MIT',
    author='Runsheng Xu',
    author_email='rxx3386@ucla.edu',
    description='v2v4real codebase',
    long_description=open("README.md").read(),
    install_requires=_read_requirements_file(),
)
