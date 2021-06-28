# -*- coding: utf-8 -*-
"""
Version File
============
*Created on 11.12.2019 by bari_is*
*For COPYING and LICENSE details, please refer to the LICENSE file*

"""
import os

try:
    filepath = os.path.dirname(os.path.realpath(__file__))

except NameError:  # We are the main py2exe script, not a module

    filepath = os.path.dirname(os.path.realpath(os.path.join(".", ".version")))

version_path = ""

for root, dirs, files in os.walk(filepath):
    if 'version.dat' in files:
        version_path = os.path.join(root, 'version.dat')

if os.path.exists(version_path):
    with open(version_path) as vfile:
        version_file = vfile.read()
        __version__ = version_file.split("\n")[-2].split(";")[-1]


else:
    print("SemPy version file not found.")
    print(filepath)
    __version__ = "0.0.0"

if __name__ == "__main__":
    print(__version__)
