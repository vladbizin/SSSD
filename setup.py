from setuptools import setup, find_packages


def readme():
  with open('README.md', 'r') as f:
    return f.read()


setup(
  name='sssd',
  version='0.0.1',
  author='Vladislav Bizin',
  author_email='vlad.bizin2001@gmail.com',
  description='Structured Space State Diffusion Model for Time Series Imputation',
  license="MIT",
  long_description=readme(),
  long_description_content_type='text/markdown',
  packages=find_packages(exclude=["tests"]),
  include_package_data=True,
  url='https://github.com/vladbizin/SSSD',
  install_requires=[
    'einops==0.7.0',
    'matplotlib==3.8.2',
    'numpy==1.26.4',
    'opt_einsum==3.3.0',
    'pandas==1.5.3',
    'pytorch_lightning==2.1.3',
    'scikit_learn==1.4.1.post1',
    'scipy==1.12.0',
    'torch==2.1.1+cu118',
    'tqdm==4.66.1',
    'tsdb==0.3.1',
  ],
  python_requires='>=3.9'
)
