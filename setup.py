import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("LICENSE", "r", encoding="utf-8") as fh:
    license_content = fh.read()

setuptools.setup(
    name="medusa-vision",
    version='0.5.3',
    author="vortezwohl",
    author_email="vortezwohl@proton.me",
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
        'opencv-python>=4.10.0.84',
        'keras>=3.4.1',
        'numpy>=1.26.4',
        'tensorflow>=2.16.2',
        'gdown>=5.2.0'
    ],
    entry_points={
        'console_scripts': [
            'medusa-clear = medusa_resources.storage.storage_manager:uninstall_resources'
        ]
    },
    include_package_data=False,
    # data_files=[
    #     (
    #         'medusa-vision-resources/weight', [
    #             'medusa-vision-resources/weight/fer2013_mini_XCEPTION_param58423_acc0.66.hdf5',
    #             'medusa-vision-resources/weight/fer2013_simple_CNN_param642935_acc0.66.hdf5',
    #             'medusa-vision-resources/weight/imdb_mini_XCEPTION_param52658_acc0.95.hdf5',
    #             'medusa-vision-resources/weight/age_vggface2_VGG16_param134674341_acc0.97.h5',
    #             'medusa-vision-resources/weight/gender_vggface2_VGG16_param134268738_acc0.97.h5',
    #             'medusa-vision-resources/weight/vggface2_VGG16_param145002878_acc0.97.h5'
    #         ]
    #     ),
    #     (
    #         'medusa-vision-resources/config', [
    #             'medusa-vision-resources/config/haarcascade_frontalface_default.xml'
    #         ]
    #     )
    # ]
)
