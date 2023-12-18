from setuptools import setup, find_packages

setup(
    name='lama-inpainting',
    version='0.2',
    packages=find_packages(),
    install_requires=[
        'pyyaml',
        'tqdm',
        'numpy',
        'easydict==1.9.0',
        'scikit-image==0.20.0',
        'scikit-learn==1.3.1',
        'opencv-python',
        'tensorflow',
        'joblib',
        'matplotlib',
        'pandas',
        'albumentations==0.5.2',
        'hydra-core==1.1.0',
        'pytorch-lightning',
        'tabulate',
        'kornia==0.5.0',
        'webdataset',
        'packaging',
        'wldhx.yadisk-direct'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    # Add the following line to generate a universal wheel file
    options={'bdist_wheel':
                 {'universal': True},
             'sdists':
                 {'formats': 'zip', 'universal': True}
             },
)