import setuptools

with open('requirements.txt', 'r') as f:
    requirements = f.readlines()

with open('readme.org', 'r') as f:
    readme = f.read()

setuptools.setup(
    name="btoms",
    version="0.0.1",
    url="https://github.com/jboes/btoms",

    author="Jacob Boes",
    author_email="jacobboes@gmail.com",

    description="Simple Bayesian optimizer for atomic structures.",
    long_description=readme,
    license='GPL-3.0',

    packages=['btoms'],
    package_dir={'btoms': 'btoms'},
    install_requires=requirements,
    include_package_data=True,
    python_requires='!=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, <4',

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Chemistry',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
