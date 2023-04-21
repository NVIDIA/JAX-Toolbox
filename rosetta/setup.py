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
    ],
    extras_require={
        'test': [
            'pytest',
        ],
        'lint': [
            'ruff',
        ],
    },
    # TODO(terry): classifiers
    keywords='machinelearning multimodal llm',
)
