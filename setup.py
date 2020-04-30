#!/usr/bin/env python

from setuptools import setup, find_packages
from sumgram import __version__

desc = """sumgram is a tool that summarizes a collection of text documents by generating the most frequent sumgrams (conjoined ngrams)"""

setup(
    name='sumgram',
    version=__version__,
    description=desc,
    long_description='See: https://github.com/oduwsdl/sumgram/',
    author='Alexander C. Nwala',
    author_email='anwala@cs.odu.edu',
    url='https://github.com/oduwsdl/sumgram/',
    packages=find_packages(),
    package_data={
        'sumgram': ['data/cleaning/*', 'data/stopwords_lists/*']
    },
    license="MIT",
    install_requires=[
        'numpy',
        'sklearn',
        'regex',
        'langdetect',
        'nltk',
        'matplotlib',
        'pandas'
    ]
)
