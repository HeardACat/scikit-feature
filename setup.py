# from distutils.core import setup
from setuptools import setup

"""
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""

NAME = "skfeature-gli"

DESCRIPTION = "Unofficial Fork of Feature Selection Repository in Python (DMML Lab@ASU)"

KEYWORDS = "Feature Selection Repository"

AUTHOR = "Jundong Li, Kewei Cheng, Suhang Wang"

AUTHOR_EMAIL = "jundong.li@asu.edu, kcheng18@asu.edu, suhang.wang@asu.edu"

MAINTAINER = "Guangyu Li"

MAINTAINER_EMAIL = "gl343@cornell.edu"

URL = "https://github.com/lguangyu/scikit-feature"

VERSION = "1.1.2"

REQUIRED = [
    "scikit-learn",
    "pandas",
    "numpy"
]

REQUIRED_CI = [
    "pytest",
    "pytest-cov",
    "black",
    "isort",
    "flake8",
    "mkdocs", "mkdocs-material",
    "mkdocstrings",
    "pytkdocs[numpy-style]"

]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
	long_description=open("README.md", "r").read(),
	long_description_content_type="text/markdown",
    keywords=KEYWORDS,
    author=AUTHOR,
    install_requires=REQUIRED,
    author_email=AUTHOR_EMAIL,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    extras_require={
        "ci": REQUIRED_CI
    },
    url=URL,
    packages=[
        "skfeature",
        "skfeature.utility",
        "skfeature.function",
        "skfeature.function.information_theoretical_based",
        "skfeature.function.similarity_based",
        "skfeature.function.sparse_learning_based",
        "skfeature.function.statistical_based",
        "skfeature.function.streaming",
        "skfeature.function.structure",
        "skfeature.function.wrapper",
    ],
)


# pushing to pypi:
# python setup.py sdist
# python setup.py bdist_wheel --universal
# twine upload dist/*
