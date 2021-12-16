import setuptools

with open("Readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="ntm_classifier",
    version="0.1.0",
    author="MACE",
    description="An externalized wrapper for classifying pngs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["ntm_classifier", "ntm_data"],
    package_dir={
        "ntm_classifier": "ntm_classifier",
        "ntm_data": "ntm_data",
        "test": "test"
    },
    python_requires=">=3.8",
    # packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={'ntm_data': ['*']},
    install_requires=requirements
)
