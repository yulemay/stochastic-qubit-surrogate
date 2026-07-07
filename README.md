# MATLAB stochastic qubit simulation

The public MATLAB code generates final ensemble-averaged Pauli expectations for a driven single qubit under coloured classical dephasing noise.

The implementation preserves the structure of the supplied one-qubit research code:

- seeded example-level randomisation;
- two Gaussian control channels;
- ten Gaussian components per channel;
- one width shared across all components in an example;
- six Pauli eigenstate preparations;
- three Pauli observables;
- Monte Carlo averaging over stochastic classical-noise realisations;
- vectorised density-matrix propagation.

The supplied `evolve_solver` helper was not part of the repository source material. For self-containment, this public implementation uses `expm(L * dt)` at each midpoint Liouvillian step.

Generate a research-scale dataset:

```matlab
generate_dataset(1000, '../data/raw')
```

Run a smaller workflow test:

```matlab
generate_dataset(10, '../data/raw', 10)
```
