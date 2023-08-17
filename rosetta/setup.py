import os
import sys
import setuptools

package_path = os.path.join(os.path.dirname(__file__), 'rosetta')
sys.path.append(package_path)
from rosetta import __version__  # noqa: E402


# Get the long description from the README file.
with open('README.md') as fp:
  _LONG_DESCRIPTION = fp.read()

setuptools.setup(
    name='rosetta',
    version=__version__,
    description='Rosetta: a Jax project for training LLM/CV/Multimodal models',
    long_description=_LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='NVIDIA',
    author_email='jax@nvidia.com',
    # TODO(terry): license, url
    packages=setuptools.find_packages(),
    package_data={
        '': ['**/*.gin'],  # not all subdirectories may have __init__.py.
    },
    scripts=[],
    install_requires=[
        'nvidia-dali-cuda120',
        'webdataset',
    ],

    extras_require={
        'test': [
            'pandas',
            'pytest',
            'pytest-xdist',
            'Pillow'
        ],
        'lint': [
            'ruff',
        ],
    },
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: GPU',
        'Environment :: GPU :: NVIDIA CUDA',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='machinelearning,multimodal,llm,jax-toolbox,jax,deep learning',
)
