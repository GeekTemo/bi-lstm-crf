# encoding=utf-8

from setuptools import setup, find_packages

setup(name='bi-lstm-crf',
      version='0.1.0',
      url='xxx',
      author='Dasheng Ji',
      author_email='jidasheng@qq.com',

      packages=find_packages(),

      include_package_data=True,
      install_requires=[
          "tqdm",
          "numpy",
          "pandas",
          "torch"
      ]
      )
