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
