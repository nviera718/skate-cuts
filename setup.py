from setuptools import setup

setup(
    name="skate-cuts",
    version="0.1.0",
    py_modules=["skate_cuts"],
    install_requires=[
        "scenedetect[opencv]>=0.6",
        "requests>=2.28",
        "Pillow>=9.0",
        "click>=8.0",
        "tqdm>=4.64",
    ],
    entry_points={
        "console_scripts": [
            "skate-cuts=skate_cuts:main",
        ],
    },
)
