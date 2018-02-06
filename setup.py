from setuptools import setup
from setuptools import find_packages

setup(
   name='game_theory',
   version='0.1',
   description='Different algorithms in game theory to finding Nash equilibria.',
   author='Marin Vlastelica',
   author_email=None,
   packages=find_packages(),
   install_requires=['blist', 'numpy'],
)