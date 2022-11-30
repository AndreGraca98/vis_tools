from setuptools import find_packages, setup

setup(
    name="vis-tools",
    version="1.0.2",
    description="vis_tools is a library created to unify the different ways to read, write and display images (opencv, matplotlib, PIL, ...)",
    author="André Graça",
    author_email="andre.p.g@sapo.pt",
    platforms="Python",
    packages=["vis"],
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
