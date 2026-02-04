"""Setup script for building the C++ shared library and Python package."""

# Note, I used Claude Code to generate some boilerplate.

import os
import subprocess
import sys

from setuptools import setup, find_packages, Distribution
from setuptools.command.build_py import build_py
from setuptools.command.editable_wheel import editable_wheel


class BinaryDistribution(Distribution):
    """Mark this as a binary distribution with platform-specific content."""

    def has_ext_modules(self):
        return True


# Since we use ctypes, we're on our own to build C++ (using ninja).
# TODO consider that my skills are decades out of date and I should learn PyBind11.
def ninja():
    subprocess.check_call(["ninja"])
    with open("compile_commands.json", "w") as compile_commands:
        subprocess.check_call(["ninja", "-t", "compdb", "cxx"], stdout=compile_commands)


class BuildPy(build_py):
    """Custom build_py that compiles .cpp to .o files, then links to shared library with ninja."""

    def run(self):
        super().run()
        ninja()


class EditableWheel(editable_wheel):
    def run(self):
        super().run()
        ninja()


setup(
    name="camspork",
    version="0.1.0",
    description='C++ Abstract Machine ("Spork") interpreter for Exo-GPU',
    author="David Zhao Akeley",
    author_email="akeley98@mit.edu",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    cmdclass={"build_py": BuildPy, "editable_wheel": EditableWheel},
    distclass=BinaryDistribution,
    python_requires=">=3.10",
    package_data={
        "camspork": ["*.so", "*.dylib", "*.dll"],
    },
)
