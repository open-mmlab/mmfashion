"""Setup script.
"""
import os
import re
import sys

from setuptools import find_packages, setup


def readme():
    """Get readme.

    Returns:
        str: readme content string.
    """
    with open('README.md') as fid:
        content = fid.read()
    return content


def get_version(version_file='mmfashion/version.py'):
    """Get version.

    Args:
        version_file (str, optional): version file path.

    Returns:
        str: version string.
    """
    with open(version_file, 'r') as fid:
        exec(compile(fid.read(), version_file, 'exec'))
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
    setup(
        name='mmfashion',
        version=get_version(),
        description='Open MMLab Fashion Toolbox',
        long_description=readme(),
        long_description_content_type='text/markdown',
        author='OpenMMLab',
        author_email='fake@email.com',
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
