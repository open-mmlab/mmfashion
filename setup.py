"""Setup script.
"""
import os
import subprocess
import time

from setuptools import find_packages, setup

MAJOR = 0
MINOR = 1
PATCH = 0
SUFFIX = ''
SHORT_VERSION = '{}.{}.{}{}'.format(MAJOR, MINOR, PATCH, SUFFIX)
VERSION_FILE = 'mmfashion/version.py'


def get_git_hash():
    """Get git hash value.

    Returns:
        str, git hash value.
    """

    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for key in ['SYSTEMROOT', 'PATH', 'HOME']:
            value = os.environ.get(key)
            if value is not None:
                env[key] = value
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        sha = out.strip().decode('ascii')
    except OSError:
        sha = 'unknown'
    return sha


def get_hash():
    """Get hash value.

    Returns:
        str, hash value.
    """
    if os.path.exists('.git'):
        sha = get_git_hash()[:7]
    elif os.path.exists(VERSION_FILE):
        try:
            from mmdet.version import __version__
            sha = __version__.split('+')[-1]
        except ImportError:
            raise ImportError('Unable to get git version')
    else:
        sha = 'unknown'
    return sha


def readme():
    """Get readme.

    Returns:
        str, readme content string.
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
short_version = '{}'
"""
    sha = get_hash()
    version = SHORT_VERSION + '+' + sha
    with open(VERSION_FILE, 'w') as fid:
        fid.write(content.format(time.asctime(), version, SHORT_VERSION))


def get_version():
    """Get version.

    Returns:
        str, version string.
    """
    with open(VERSION_FILE, 'r') as fid:
        exec compile(fid.read(), VERSION_FILE, 'exec')
    return locals()['__version__']


if __name__ == '__main__':
    write_version_py()
    setup(
        name='mmfashion',
        version=get_version(),
        description='Open MMLab Fashion Toolbox',
        long_description=readme(),
        author='OpenMMLab',
        author_email='https://github.com/open-mmlab/mmfashion',
        keywords='computer vision, fashion',
        url='https://github.com/open-mmlab/mmfashion',
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
        ],
        license='Apache License 2.0',
        setup_requires=['pytest-runner'],
        tests_require=['pytest'],
        install_requires=[
            'mmcv', 'numpy', 'scikit-image', 'pandas', 'torch', 'torchvision'
        ],
        packages=find_packages(),
        zip_safe=False)
