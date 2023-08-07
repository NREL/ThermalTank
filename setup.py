from pathlib import Path

from setuptools import setup, find_packages

from thermaltank import VERSION

readme_file = Path(__file__).parent.resolve() / 'README.md'
readme_contents = readme_file.read_text()

setup(
    name="ThermalTank",
    author="Matt Mitchell",
    license='MIT',
    long_description=readme_contents,
    version=VERSION,
    packages=find_packages(exclude=["test", "tests", "test.*"]),
    long_description_content_type='text/markdown',
    python_requires=">=3.8"
)
