"""Setup script.
"""
import os
import re
import subprocess
import sys
import time

from setuptools import find_packages, setup

MAJOR = 0
MINOR = 4
PATCH = 0
SUFFIX = ''
SHORT_VERSION = '{}.{}.{}{}'.format(MAJOR, MINOR, PATCH, SUFFIX)
VERSION_FILE = 'mmfashion/version.py'


def get_git_hash():
    """Get git hash value.

    Returns:
        str: git hash value.
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
        str: hash value.

    Raises:
        ImportError: import error.
    """
    if os.path.exists('.git'):
        sha = get_git_hash()[:7]
    elif os.path.exists(VERSION_FILE):
        try:
            from mmfashion.version import __version__
            sha = __version__.split('+')[-1]
        except ImportError:
            raise ImportError('Unable to get git version')
    else:
        sha = 'unknown'
    return sha


def readme():
    """Get readme.

    Returns:
        str: readme content string.
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
        str: version string.
    """
    with open(VERSION_FILE, 'r') as fid:
        exec(compile(fid.read(), VERSION_FILE, 'exec'))
    return locals()['__version__']


def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
        specific versioning information.

    Args:
        fname (str): path to requirements file.
        with_version (bool, default=False): if True include version specs.

    Returns:
        List[str]: list of requirements items.

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file.
        """
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as fid:
            for line in fid.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if os.path.exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


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
        setup_requires=parse_requirements('requirements/build.txt'),
        install_requires=parse_requirements('requirements/runtime.txt'),
        tests_require=parse_requirements('requirements/tests.txt'),
        extras_require={
            'all': parse_requirements('requirements.txt'),
            'build': parse_requirements('requirements/build.txt'),
            'tests': parse_requirements('requirements/tests.txt')
        },
        packages=find_packages(),
        zip_safe=False)
