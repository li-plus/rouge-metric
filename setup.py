import os
import re

import setuptools

here = os.path.dirname(__file__)

# get the version string
with open(os.path.join(here, 'rouge_metric', '__init__.py')) as f:
    version = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)

setuptools.setup(
    name='rouge-metric',
    version=version,
    author='Jiahao Li',
    author_email='liplus17@163.com',
    maintainer='Jiahao Li',
    maintainer_email='liplus17@163.com',
    url='https://github.com/li-plus/rouge-metric',
    description='A fast python implementation of full ROUGE metrics for automatic summarization.',
    long_description=open(os.path.join(here, 'README.md')).read(),
    long_description_content_type='text/markdown',
    # download_url=''
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Text Processing :: Linguistic',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
    ],
    # platforms=[],
    keywords=['rouge', 'summarization', 'natural language processing',
              'computational linguistics'],
    license='MIT',
    python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*',
    install_requires=[
        'typing;python_version<"3.5"'
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-cov',
            'flake8',
        ]
    },
    include_package_data=True,
    entry_points={
        'console_scripts': ['rouge-metric=bin.rouge:main'],
    },
)
