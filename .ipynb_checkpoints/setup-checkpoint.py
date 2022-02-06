# setup.py
from setuptools import setup, find_packages

setup(name='emotionpredictor',
      version='0.1',
      description='EmotionPredictor: Picture affect prediction',
      url='https://github.com/robinszym/EmotionPredictor',
      author='Robin Szymczak',
      packages=find_packages(),
      install_requires=['torch',
                        'torchvision',
                        'scikit-learn',
                        'pandas',
                        'matplotlib',
                        'plotly',
                        'seaborn',
                        'Pillow',
                        'jupyter',
                        'tqdm',
                        'termcolor',],
      python_requires='>=3')