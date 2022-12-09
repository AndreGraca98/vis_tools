import json

from setuptools import find_packages, setup

setup(
    name="vis-tools",
    version=json.load(open("vis/version.json"))["version"],
    description="vis_tools is a library created to unify the different ways to read, write and display images (opencv, matplotlib, PIL, ...)",
    author="André Graça",
    author_email="andre.p.g@sapo.pt",
    platforms="Python",
    packages=["vis_tools"],
    install_requires=[
        "pandas",
        "numpy",
        "einops",
        "matplotlib",
        "opencv-python",
        "Pillow>=5.2",
        "imageio",
        "easydict",
        "tqdm",
    ],
)
