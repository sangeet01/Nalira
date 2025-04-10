from setuptools import setup, find_packages

setup(
    name="Nalira",
    version="0.1.0",
    description="A hybrid generative AI pipeline for natural product optimization and receptor prediction",
    author="Sangeet S., Bipindra Pandey",
    author_email="your.email@example.com",  # Replace with your email
    license="Apache-2.0",
    packages=find_packages(),
    install_requires=[
        "rdkit",          # SMILES validation, fingerprints
        "openbabel",      # SMILES standardization
        "pubchempy",      # PubChem API fetches
        "pandas",         # .csv batch processing
        "numpy",          # General array ops
        "torch",          # REINVENT RNN, Deep DTI
        "autodock-vina",  # Optional docking
    ],
    python_requires=">=3.8",  # Modern Python for compatibility
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "nalira = nalira:main",  # Assumes nalira.py has a main() function
        ]
    },
)