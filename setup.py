"""Package the portable runtime only; research modules remain in the Git repository."""
from setuptools import setup
from setuptools.command.build_py import build_py

RUNTIME = {'__init__', '__main__', 'api', 'audio', 'frontend', 'normalization',
           'normalization_v2', 'onnx_runtime'}


class RuntimeBuild(build_py):
    def find_package_modules(self, package, package_dir):
        return [entry for entry in super().find_package_modules(package, package_dir)
                if entry[1] in RUNTIME]


setup(cmdclass={'build_py': RuntimeBuild})
