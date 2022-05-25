![GitHub](https://img.shields.io/github/license/pyubero/quantitative_mutations)

# Quantitative mutations using `qmut`
The package qmut contains mainly a class called QuantitativeMutation to enable the simulation of quantitative mutations of whole genome metabolic reconstructions. 

## Installation
 1. Check that you have `cobrapy` installed in your environment
 2. Download and copy the `\cobrapy_qmut` folder in your working directory.


## Tutorial
In this tutorial we will show the main applications and methods of the QuantiativeMutation objects, briefly:


1. Load a metabolic model and create its QuantitativeMutation object
2. Using `Q.optimize()` and `Q.slim_optimize()` to find solutions and biomass production rate, respectively.
3. Generating random media following the protocol of [Wang and Zhang](https://pubmed.ncbi.nlm.nih.gov/19132081/) and applying it to `Q`.
4. Compute maximal bounds of `Q` across a list of media. And save/load them with `Q.load_bounds()` and `Q.save_bounds()`
5. Check how gene reaction rules are quantitatively interpreted.
6. Working with relative gene dosages and quantitative mutations.
7. Compute individuals, and populations in a single or in multiple processors.


## Reference
If you find this class useful and you use it in your work, please *also* cite its paper:
The limitations of phenotype prediction in metabolism, Pablo Yubero, Alvar A. Lavin, Juan F. Poyatos
bioRxiv 2022.05.19.492732; doi: https://doi.org/10.1101/2022.05.19.492732

## Contact
If your have any issue with this repo please contact me through github rather than email. 
Feel free to ask for updated versions of it as perhaps I miss to keep this repo up to date.
