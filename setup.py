from setuptools import setup

setup(name='COSIpy',
      version='0.1',
      description='Analysing COSI data with Stan model',
      url='https://github.com/tsiegert/COSIpy',
      author='Thomas Siegert',
      author_email='tsiegert@ucsd.edu',
      #license='MIT',
      packages=['COSIpy'],
      install_requires=[
          'numpy',
          'pystan',
          'matplotlib',
          'tqdm',
          'pandas',
          'shapely',
          'pystan'
      ],
      zip_safe=False)
