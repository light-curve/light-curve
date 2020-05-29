from setuptools import find_packages, setup
from setuptools_rust import RustExtension


setup_requires = ['setuptools-rust>=0.10.2']
install_requires = ['numpy']
# test_requires = install_requires + ['pytest']

setup(
    name='light-curve',
    version='0.1.0',
    description='Extract features from light curve',
    rust_extensions=[RustExtension(
        'light_curve.feature',
        './Cargo.toml',
    )],
    install_requires=install_requires,
    setup_requires=setup_requires,
    # test_requires=test_requires,
    packages=find_packages(),
    zip_safe=False,
)
