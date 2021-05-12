import codecs
from pathlib import Path

from setuptools import setup, find_packages

from src import VERSION

this_dir = Path(__file__).parent
with codecs.open(str(this_dir / "README.md"), encoding="utf-8") as ld_file:
    long_description = ld_file.read()

with codecs.open(str(this_dir / "LICENSE"), encoding="utf-8") as lic_file:
    project_license = lic_file.read()

setup(
    name="ThermalTank",
    author="Matt Mitchell",
    license=project_license,
    long_description=long_description,
    version=VERSION,
    packages=find_packages(exclude=["test", "tests", "test.*"]),
    long_description_content_type='text/markdown',
    python_requires=">=3.7"
)
