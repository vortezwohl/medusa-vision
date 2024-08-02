import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("LICENSE", "r", encoding="utf-8") as fh:
    license_content = fh.read()

setuptools.setup(
    name="medusa-vision",
    version="0.1.0",
    author="vortezwohl",
    author_email="2310108909@qq.com",
    description="Emotion and gender recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vortezwohl/medusa-vision",
    project_urls={
        "Bug Tracker": "https://github.com/vortezwohl/medusa-vision/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
    package_dir={"": "medusa"},
    packages=setuptools.find_packages(where="medusa"),
    python_requires=">=3.6",
    license=license_content
)
