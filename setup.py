"""Setup script.
"""
import time
from setuptools import setup

MAJOR = 0
MINOR = 1
PATCH = 0
SUFFIX = ''
SHORT_VERSION = '{}.{}.{}{}'.format(MAJOR, MINOR, PATCH, SUFFIX)
VERSION_FILE = 'mmfashion/version.py'


def get_version():
    """Get version.

    Returns:
        Version string.
    """
    with open(VERSION_FILE, 'r') as fid:
        exec (compile(fid.read(), VERSION_FILE, 'exec'))
    return locals()['__version__']


def readme():
    """Get readme.

    Returns:
        Readme content string.
    """
    with open('README.md') as fid:
        content = fid.read()
    return content


def write_version_py():
    """Write version.py.
    """
    content = """# GENERATED VERSION FILE
# TIME: {}
__version__ = '{}'
"""
    with open(VERSION_FILE, 'w') as fid:
        fid.write(content.format(time.asctime(), SHORT_VERSION))


if __name__ == '__main__':
    write_version_py()
    setup(
        name='mmfashion',
        version=get_version(),
        description='Open MMLab Fashion Toolbox',
        long_description=readme(),
        keywords='computer vision, fashion',
        url='https://github.com/open-mmlab/mmfashion',
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
        ],
        license='GPLv3',
        setup_requires=['pytest-runner'],
        tests_require=['pytest'],
        install_requires=['mmcv', 'numpy', 'torch', 'torchvision'],
        zip_safe=False)
