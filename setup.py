from pathlib import Path
from setuptools import find_namespace_packages, setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

# Define our package
setup(
    name="",
    version=0.1,
    description="",
    author="Miko Planas",
    author_email="miksbon@gmail.com",
    python_requires=">=3.9",
    packages=find_namespace_packages(),
    install_requires=[required_packages],
)
