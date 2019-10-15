from setuptools import setup

setup(name='Neuro_Shit',
      version='0.1',
      description='Yep',
      url='http://github.com/sahahn/Neuro_Shit',
      author='Sage Hahn',
      author_email='sahahn@euvm.edu',
      license='MIT',
      packages=['Neuro_Shit'],
      install_requires=[
          'nibabel',
          'pandas',
          'nilearn',
          'numpy',
      ],
      zip_safe=False)
