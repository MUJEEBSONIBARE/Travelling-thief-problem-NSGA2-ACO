# Mapped Distance Matrices

This folder contains the distance-matrix data used in the Travelling Thief Problem (TTP) experiments.

## Contents

The distance matrices store pre-computed distances between locations used by the optimisation algorithms. They support route evaluation and optimisation during the experiments.

The original `.dat` files are retained locally but are not included in the GitHub repository where individual files exceed GitHub's standard web-upload limit.

## Purpose

The distance matrices are used to:

- provide pre-computed distances between problem locations;
- support route-distance calculations;
- evaluate candidate solutions during optimisation; and
- avoid repeatedly calculating the same pairwise distances during experiments.

## Data Availability

The complete distance-matrix files are available with the original project files and can be used to reproduce the experiments.

For repository portability, the large matrix files are excluded from this GitHub repository. The project source code and supporting documentation remain available in the repository.

## Reproducibility

To reproduce the experiments, obtain the corresponding distance-matrix files and place them in this directory before running the relevant scripts.

Expected structure:

```text
Mapped_distance_matrix/
├── [distance matrix .dat files]
└── README.md
```

## Project Context

These matrices form part of a Travelling Thief Problem optimisation project combining route optimisation and item-selection decisions using metaheuristic approaches such as Ant Colony Optimisation (ACO) and NSGA-II.
