__author__ = "Jon Hollenbach"
__copyright__ = "Copyright Jon Hollenbach (2024)"
__maintainer__ = "Jon Hollenbach"
__email__ = "jhollen3@jhu.edu"
__version__ = 0.1

from setuptools import setup, find_packages
import os

module_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    setup(
        name='logicalEELS',
        python_requires='>=3.8',
        version=__version__,
        description='TensorFlow Framework for classification of EELS data with embedded theoretical structures',
        long_description=open(os.path.join(module_dir, 'README.md')).read(),
        long_description_content_type='text/markdown',
        url='https://github.com/hollejd1/logicalEELS/',
        author='Jon Hollenbach',
        author_email='jhollen3@jhu.edu',
        license='JHU Academic Software License',
        packages=find_packages(),
        zip_safe=False,
        install_requires=[
            'tensorflow==2.11.0',
            'numpy',
            'hyperspy',
            'scikit-learn',
        ],
        classifiers=['Programming Language :: Python',
                     'Development Status :: 1 - Alpha',
                     'Intended Audience :: Science/Research',
                     'Operating System :: OS Independent',
                     'Topic :: Scientific/Engineering']
    )