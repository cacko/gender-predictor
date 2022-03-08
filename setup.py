from setuptools import setup, find_packages

setup(name='gender_predictor',
      version='1.0.2',
      description=(' '),
      author='Clint Valentine',
      author_email='valentine.clint@gmail.com',
      url='https://github.com/clintval/gender_predictor',
      package_data={'': ['names.pickle']},
      packages=find_packages(),
      install_requires=['nltk'],
      )
