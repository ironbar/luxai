import os
from setuptools import setup, find_packages

#Version of the software
from luxai import __version__

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="luxai",
    version=__version__,
    author="ironbar",
    author_email="guillermobarbadillo@gmail.com",
    description=("Gather the most resources and survive the night!"),
    license='All rights reserved',
    long_description=read('README.md'),
    classifiers=[],
    packages=find_packages(exclude=['notebooks', 'reports', 'tests',
                                    'logs', 'forum', 'data']),
    include_package_data=True,
    zip_safe=False,
    setup_requires=['pytest-runner',],
    tests_require=['pytest',],
    test_suite='tests',
)
