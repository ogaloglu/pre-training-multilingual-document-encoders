from setuptools import find_packages, setup
from typing import List


def get_requirements(file_path: str) -> List[str]:
    """This function will return the list of requirements.

    Args:
        file_path (str): Path of the requirements.

    Returns:
        List[str]: List of requirements.
    """
    requirements = list()
    with open(file_path) as f:
        requirements = f.read().splitlines()

    return requirements


setup(
    name="multilingual_document_encoder",
    version="0.0.1",
    author="Onur Galoglu",
    author_email="onur.galoglu@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
