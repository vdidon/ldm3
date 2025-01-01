from distutils.core import setup
from setuptools import setup


with open("README.md", "r") as fh:
    long_description = fh.read()
setup(
    name="ldm",
    version="0.1.4",
    description="get the landmars,positions,features for reconition of the face in the image",
    long_description=long_description,
    long_description_content_type="text/markdown",

    license = "MIT",
    author="vdidon",
    author_email="vdidon@live.fr",
    url="https://github.com/vdidon/ldm3",
    py_modules=['ldm'],
    platforms = 'any'
)
