# setup.py COPYRIGHT Fujitsu Limited 2021
import os
import sys
import platform
from setuptools import setup

def _requires_from_file(filename):
    return open(filename).read().splitlines()

python_min_version = (3, 7, 0)
python_min_version_str = '.'.join(map(str, python_min_version))
if sys.version_info < python_min_version:
    print("You are using Python {}. Python >={} is required.".format(platform.python_version(),
                                                                     python_min_version_str))
    sys.exit(-1)

cwd = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(cwd, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='cac',
    version='1.0.0',
    url='https://github.com/FujitsuLaboratories/CAC',
    author='Yasufumi Sakai',
    author_email='sakaiyasufumi@fujitsu.com',
    maintainer='Akihiko Kasagi',
    maintainer_email='kasagi.akihiko@fujitsu.com',
    description='Content-Aware Computing library',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['cac', 'cac.gradskip',  'cac.relaxed_sync', 'cac.pruning'],
    install_requires=_requires_from_file('requirements.txt'),
    python_requires='>={}'.format(python_min_version_str),
    license="See https://github.com/FujitsuLaboratories/CAC/LICENSE",
    classifiers=[
        'Development Status :: 1 - Planning',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ]
)
