import subprocess
import os
from setuptools import setup, find_packages

# Initialize and update Git submodule
subprocess.run(["git", "submodule", "init"])
subprocess.run(["git", "submodule", "update"])
# Change directory to submodule
os.chdir("src/visio_gptq/GPTQ-for-LLaMa")
# Checkout cuda branch
subprocess.run(["git", "checkout", "81fe867ae302a057a525e8584c6b927a1ba7b748"])
# Install requirements
subprocess.run(["pip", "install", "-r", "requirements.txt"])
# Install package
subprocess.run(["python", "setup_cuda.py", "install"])
# Change directory back to root
os.chdir("../../..")

DISTNAME = "visio_gptq"
DESCRIPTION = "GPTQ implementation for Visio"
PYTHON_VERSION = ">=3.10"
AUTHOR = "Marco João, Anderson Cançado, Frederico Oliveira"
PROJECT_URLS = {
    "Bug Tracker": "https://github.com/Gaspard-Bruno/visio-mlops/issues",
    "Source Code": "https://github.com/Gaspard-Bruno/visio-mlops",
}
KEYWORDS = "machine learning, llm, gptq, visio, ml"
VERSION = "0.0.0"
with open(f"src/{DISTNAME}/version.py", encoding="utf-8") as fid:
    for line in fid:
        if line.startswith("__version__"):
            VERSION = line.strip().split()[-1][1:-1]
            break

with open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()


def __get_requirements():
    return required

setup(
    name=DISTNAME,
    version=VERSION,
    python_requires=PYTHON_VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    project_urls=PROJECT_URLS,
    keywords=KEYWORDS,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=__get_requirements(),
    include_package_data=True
)
