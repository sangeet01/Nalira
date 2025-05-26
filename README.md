# Nalira: Natural Product Optimization & Receptor Prediction Pipeline

Nalira is a hybrid AI pipeline for optimizing natural products and predicting receptors. It leverages machine learning to optimize properties such as solubility, lipophilicity, and synthetic accessibility while predicting receptor interactions. Built on REINVENT (RNN + RL) and Deep DTI frameworks, Nalira provides an end-to-end solution for natural product development and drug discovery applications.

## Features

- **Input**: SMILES, molecular descriptors, receptor data
- **Output**: Optimized natural products, predicted receptor interactions, property scores
- **Pipeline**: Property prediction, optimization, receptor binding analysis, validation
- **Optimization**: Support for local CPU or distributed computing

## Installation

1. Clone the repository:

```
git clone https://github.com/sangeet-sangit/Nalira.git
cd Nalira
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Configure the environment in config.py (optional)

## Usage

Run the pipeline with default settings:

```
python nalira.py --input smiles.txt --output-dir ./results
```

## Arguments

- `--input`: Path to input file containing SMILES strings
- `--model`: Select optimization model (default: reinvent)
- `--receptor`: Target receptor for optimization (optional)
- `--output-dir`: Directory for results (default: ./results)

## Training

### REINVENT Model Training

```python
# Train REINVENT model on custom dataset
from nalira import train_model
train_model(data_path="./data", epochs=100, device="cuda")
```

### Deep DTI Training

```python
# Train DTI model for receptor prediction
from nalira import train_dti
train_dti(protein_data="./proteins.csv", compound_data="./compounds.smi")
```

## Testing

- Test with sample molecules from public databases
- Validate receptor predictions against experimental data
- Check results in the output directory

## Dependencies

- Python 3.7+
- PyTorch
- RDKit
- scikit-learn
- NumPy, Pandas

## Initialize Git and Push

```
cd Nalira
git init
git add .
git commit -m "Initial commit of Nalira pipeline"
git branch -M main
git remote add origin https://github.com/Sangeet01/Nalira.git
git push -u origin main
```

## Contact

For issues or contributions, please contact:
- GitHub: [@Sangeet01](https://github.com/Sangeet01)

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## PS

This is a work in progress. Expect bugs and incomplete functionality in this early release. Contributions, bug reports, and feature requests are welcome!

## PSS
Sangeet's the name, a daft undergrad splashing through chemistry and code like a toddler—my titrations are a mess, and I've used my mouth to pipette.
