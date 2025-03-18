from setuptools import setup, find_packages
import os

from setuptools import setup, find_packages
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Get version from environment or fallback to default
version = os.environ.get("VERSION_PLACEHOLDER", "1.0.0")
print(f"Version: {version}")

setup(
    name='shaining',
    version=version,
    description='SHAining a light on process mining benchmarks',
    license='MIT',
    url='https://github.com/lmu-dbs/shampu.git',
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.9',  # Python version compatibility
    install_requires=[
        'shap>=0.45.0,<1.0.0',
        'feeed>=1.2.0,<2.0.0',
        'imblearn==0.0',
        'matplotlib>=3.8.4,<4.0.0',
        'numpy>=1.26.4,<2.0.0',
        'pandas>=2.2.2,<3.0.0',
        'pm4py>=2.7.2,<3.0.0',
        'scikit-learn>=1.2.2,<2.0.0',
        'scipy>=1.13.0,<2.0.0',
        'seaborn>=0.13.2,<0.14.0',
        'smac>=2.0.2,<3.0.0',
        'tqdm>=4.65.0,<5.0.0',
        'gedi>=1.0.8,<2.0.0',
        'pytest~=8.3.3',
        'tabulate>=0.8.9',
        'func-timeout>=4.3.5',
    ],
    packages=find_packages(),  # Automatically find packages
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)