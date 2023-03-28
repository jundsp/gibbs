from setuptools import setup, find_packages

setup(name='gibbs',
    version='0.0',
    description='Gibbs samplers for inference and learning of hierarchical Bayesian models.',
    author='Julian Neri',
    author_email='julian.neri@mcgill.ca',
    url='https://www.github.com/jundsp/gibbs',
    license='GNU',
    packages=find_packages(),
    include_package_data=True,
    package_dir={'gibbs': 'gibbs'},
    package_data={'gibbs': ['mplstyles/*.mplstyle']},
    install_requires=['matplotlib',
                      'numpy',
                      'scipy',
                      'tqdm']
    )