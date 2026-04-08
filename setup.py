from setuptools import setup

setup(
    name="skate-cuts",
    version="0.2.0",
    py_modules=["skate_cuts"],
    install_requires=[
        "scenedetect[opencv]>=0.6",
        "click>=8.0",
        "tqdm>=4.64",
    ],
    entry_points={
        "console_scripts": [
            "skate-cuts=skate_cuts:main",
        ],
    },
)
