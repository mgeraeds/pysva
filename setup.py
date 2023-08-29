from setuptools import setup, find_packages 

with open("requirements.txt") as f:
      requirements = f.read().splitlines()

setup(name='pysva',
      version='0.1.0',
      author='Marlein Geraeds',
      author_email='m.e.g.geraeds@tudelft.nl',
      packages=find_packages(), 
      install_requires=requirements)
