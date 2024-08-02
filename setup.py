import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("LICENSE", "r", encoding="utf-8") as fh:
    license_content = fh.read()

setuptools.setup(
    name="medusa-vision",
    version='0.2.6',
    author="vortezwohl",
    author_email="2310108909@qq.com",
    description="Emotion and gender recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license=license_content,
    url="https://github.com/vortezwohl/medusa-vision",
    project_urls={
        "Bug Tracker": "https://github.com/vortezwohl/medusa-vision/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.6",
    install_requires=[
        'opencv-python==4.10.0.84',
        'keras==3.4.1',
        'numpy==1.26.4',
        'tensorflow==2.16.2'
    ],
    include_package_data=True,
    data_files=[
        (
            'medusa-vision-static/weight', [
                'medusa-vision-static/weight/fer2013_mini_XCEPTION_param58423_epoch102_acc0.66.hdf5',
                'medusa-vision-static/weight/fer2013_simple_CNN_param642935_epoch985_acc0.66.hdf5',
                'medusa-vision-static/weight/imdb_mini_XCEPTION_param52658_epoch21_acc0.95.hdf5'
            ]
        ),
        (
            'medusa-vision-static/config', [
                'medusa-vision-static/config/haarcascade_frontalface_default.xml'
            ]
        )
    ]
)
