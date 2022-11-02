# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import platform
import shutil
import sys
import warnings
from setuptools import find_packages, setup


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


version_file = 'mmseg/version.py'


def get_version():
    with open(version_file) as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import re
    import sys
    from os.path import exists
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
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
        with open(fpath) as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    yield from parse_line(line)

    def gen_packages_items():
        if exists(require_fpath):
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


def add_mim_extension():
    """Add extra files that are required to support MIM into the package.

    These files will be added by creating a symlink to the originals if the
    package is installed in `editable` mode (e.g. pip install -e .), or by
    copying from the originals otherwise.
    """

    # parse installment mode
    if 'develop' in sys.argv:
        # installed by `pip install -e .`
        if platform.system() == 'Windows':
            # set `copy` mode here since symlink fails on Windows.
            mode = 'copy'
        else:
            mode = 'symlink'
    elif 'sdist' in sys.argv or 'bdist_wheel' in sys.argv or \
            platform.system() == 'Windows':
        # installed by `pip install .`
        # or create source distribution by `python setup.py sdist`
        # set `copy` mode here since symlink fails with WinError on Windows.
        mode = 'copy'
    else:
        return

    filenames = ['tools', 'configs', 'model-index.yml']
    repo_path = osp.dirname(__file__)
    mim_path = osp.join(repo_path, 'mmseg', '.mim')
    os.makedirs(mim_path, exist_ok=True)

    for filename in filenames:
        if osp.exists(filename):
            src_path = osp.join(repo_path, filename)
            tar_path = osp.join(mim_path, filename)

            if osp.isfile(tar_path) or osp.islink(tar_path):
                os.remove(tar_path)
            elif osp.isdir(tar_path):
                shutil.rmtree(tar_path)

            if mode == 'symlink':
                src_relpath = osp.relpath(src_path, osp.dirname(tar_path))
                try:
                    os.symlink(src_relpath, tar_path)
                except OSError:
                    # Creating a symbolic link on windows may raise an
                    # `OSError: [WinError 1314]` due to privilege. If
                    # the error happens, the src file will be copied
                    mode = 'copy'
                    warnings.warn(
                        f'Failed to create a symbolic link for {src_relpath}, '
                        f'and it will be copied to {tar_path}')
                else:
                    continue

            if mode == 'copy':
                if osp.isfile(src_path):
                    shutil.copyfile(src_path, tar_path)
                elif osp.isdir(src_path):
                    shutil.copytree(src_path, tar_path)
                else:
                    warnings.warn(f'Cannot copy file {src_path}.')
            else:
                raise ValueError(f'Invalid mode {mode}')


if __name__ == '__main__':
    add_mim_extension()
    setup(
        name='mmsegmentation',
        version=get_version(),
        description='Open MMLab Semantic Segmentation Toolbox and Benchmark',
        long_description=readme(),
        long_description_content_type='text/markdown',
        author='MMSegmentation Contributors',
        author_email='openmmlab@gmail.com',
        keywords='computer vision, semantic segmentation',
        url='http://github.com/open-mmlab/mmsegmentation',
        packages=find_packages(exclude=('configs', 'tools', 'demo')),
        include_package_data=True,
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
        ],
        license='Apache License 2.0',
        install_requires=parse_requirements('requirements/runtime.txt'),
        extras_require={
            'all': parse_requirements('requirements.txt'),
            'tests': parse_requirements('requirements/tests.txt'),
            'build': parse_requirements('requirements/build.txt'),
            'optional': parse_requirements('requirements/optional.txt'),
            'mim': parse_requirements('requirements/mminstall.txt'),
        },
        ext_modules=[],
        zip_safe=False)
