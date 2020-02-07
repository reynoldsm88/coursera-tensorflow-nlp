#!/usr/bin/env python3

from setuptools import setup, find_packages

setup( name = 'tensorflow-nlp',
       version = '1.0',
       py_modules = [ 'tensorflow-nlp' ],
       packages = find_packages( exclude = ("docs", "tests") ),
       test_suite = "tests",
       tests_require = [ 'nose' ] )