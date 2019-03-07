from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='representation_learning_for_transcriptomics',
      version='0.1',
      description='unlearn.representation_learning_for_transcriptomics',
      url='http://github.com/unlearnai/representation_learning_for_transcriptomics.git',
      author='Unlearn.AI',
      author_email='info@unlearn.ai',
      license='MIT',
      packages=['representation_learning_for_transcriptomics'],
      install_requires=[
          'numpy > 1.15',
          'pandas > 0.23',
          'scikit-learn > 0.20',
          'lifelines > 0.14'
          ],
      tests_require=[
          'pytest'
      ],
      zip_safe=False)
