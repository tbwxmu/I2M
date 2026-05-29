from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    """Force wheel tagging as platform-specific because the package ships a prebuilt .so."""

    def has_ext_modules(self):
        return True


setup(distclass=BinaryDistribution)
