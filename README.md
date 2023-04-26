# HERACLES
![License: Apache-2.0](https://img.shields.io/github/license/gil2rok/heracles)

HERACLES algorithm reconstructs CRISPR-Cas9 phylogenetic lineages with hyperbolic embeddings.

**Note: this repo is under active development and is not yet fully functional**

[![HERACLES](https://images.fineartamerica.com/images/artworkimages/mediumlarge/3/crispr-genetic-science-dna-substitution-drawing-frank-ramspott.jpg)](img)

## Overview

Insert CRISPR-Cas9 casettes (i.e. cellular "barcodes") into a cell and sequence its descendants to reconstruct the evolutionary cell lineage. Using a CRISPR-specific continous time Markov chain, model the mutations that accumulate on these casettes. Then construct a function to quanitify the likelihood of a particular evolutionary tree, in both tree topology and branch lengths. Approixmate the tree-metric by embedding points in hyperbolic space. Lastly, optimize over the hyperbolic embeddings using Riemannian stochastic gradient descent.

**In plain English:** HERACLES algorithm uses CRISPR gene editing :dna: to insert a "barcode" into a cell. After hundreds of generations of evolution, the cell's descendants will have accumulated mutations on this barcode .

We take a novel approach that involves a bunch of crazy math -- continous time Markov chains, hyperbolic geometry, and gradient-based optimization  -- that allows us to reconstruct the evolutionary tree from these modified barcodes.