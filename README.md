 # Stochastic Qubit Surrogate

This repository contains a single-qubit simulation and black-box recurrent model.

The workflow starts in MATLAB, where stochastic quantum-response data are generated. The MATLAB files are then processed in Python and used to train a TensorFlow GRU model. Once the GRU is trained, it can also be frozen and used as a differentiable surrogate for pulse optimisation.

The main workflow is:


MATLAB simulation
        |
        v
dataset_XXXX.mat
        |
        v
Python post-processing
        |
        v
GRU-ready pulse sequences
        |
        v
TensorFlow black-box GRU
        |
        v
Predicted final measurements
        |
        v
Optional pulse controller
```

The model in this repository is black-box. It only learns the map between the pulse parameters and the final measurement outcomes. The Hamiltonian, Liouvillian and noise operator are not passed to the recurrent model.

---

## Getting started

Before starting, make sure the following are installed:

- [Git](https://git-scm.com/)
- [Python 3.11](https://www.python.org/)
- MATLAB
- A code editor such as [Visual Studio Code](https://code.visualstudio.com/)

You can check Git and Python from a terminal:

```bash
git --version
python --version
```

### Clone the repository

Create or move to the folder where you want the project, then run:

```bash
git clone git@github.com:yulemay/stochastic-qubit-surrogate.git
cd stochastic-qubit-surrogate
```

If SSH is not configured for GitHub, the HTTPS clone command can be used instead:

```bash
git clone https://github.com/yulemay/stochastic-qubit-surrogate.git
cd stochastic-qubit-surrogate
```

### Create the Python environment

For Mac or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,qutip]"
```

For Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,qutip]"
```

The package contains the TensorFlow model, the dataset-processing code and the controller code. QuTiP is included as an optional demonstration dependency.

---

## Repository structure

The main folders are:

```text
stochastic-qubit-surrogate/
├── matlab/
│   ├── generate_dataset.m
│   ├── simulate_qubit_example.m
│   └── generate_arbitrary_noise.m
│
├── python/
│   ├── build_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── optimise_control.py
│
├── src/
│   └── qubit_surrogate/
│       ├── controller.py
│       ├── data.py
│       ├── metrics.py
│       ├── ml_model.py
│       ├── plotting.py
│       └── quantum_targets.py
│
├── examples/
│   └── qutip_coloured_dephasing_demo.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── results/
│   ├── figures/
│   ├── models/
│   └── control/
│
├── tests/
├── config/
└── docs/
```

The easiest way to use the repository is to follow the sections below in order.

---

## 1. Generate the single-qubit simulation data

The simulation is written in MATLAB.

Open MATLAB and move into the `matlab` folder:

```matlab
cd matlab
```

To generate the full dataset:

```matlab
generate_dataset(1000, '../data/raw')
```

This generates 1000 pulse-response examples. By default, each example is averaged over 1000 stochastic noise realisations.

For a quick pipeline test, use a smaller dataset and fewer noise realisations:

```matlab
generate_dataset(10, '../data/raw', 10)
```

The first argument is the number of pulse examples. The third argument is the number of stochastic realisations, `K`.

### What is simulated?

The qubit is driven by two control channels:

```text
x control
y control
```

Each channel is constructed from ten Gaussian components.

For one example, MATLAB stores the pulse parameters as:

```text
pulse_parameters.shape = (2, 10, 3)
```

The three values stored for each Gaussian are:

```text
[amplitude, centre, width]
```

The simulation prepares the six Pauli eigenstates in the following order:

```text
+X
-X
+Y
-Y
+Z
-Z
```

For each initial state, the final expectations of the three Pauli observables are calculated:

```text
<X>
<Y>
<Z>
```

Therefore, each simulation produces:

```text
6 initial states x 3 observables = 18 outputs
```

The output of each example is saved as:

```text
dataset_0001.mat
dataset_0002.mat
dataset_0003.mat
...
```

Each `.mat` file contains:

```text
pulse_parameters
data
```

where `data` is the 18-component ensemble-averaged response.

### Classical noise

The MATLAB simulation generates coloured classical dephasing noise using random spectral phases. The resulting stochastic trace is coupled through the qubit `sigma_z` operator.

For each noise realisation, the density matrices are propagated independently. The final Pauli expectations are then averaged over all `K` realisations.

The relevant MATLAB files are:

```text
matlab/simulate_qubit_example.m
matlab/generate_arbitrary_noise.m
```

---

## 2. Process the MATLAB files for the recurrent model

The raw MATLAB pulse tensor is not passed directly to TensorFlow.

The GRU expects an ordered sequence. The purpose of `build_dataset.py` is to convert each MATLAB example into this sequence representation.

Run:

```bash
python python/build_dataset.py \
    --mat-folder data/raw \
    --out-file data/processed/qubit_response_dataset.npz
```

### What does the post-processing do?

The MATLAB pulse tensor has shape:

```text
(2, 10, 3)
```

The Python code rearranges it into:

```text
(10, 6)
```

For each of the ten Gaussian positions, the six input features are:

```text
A_x
tau_x
sigma_x
A_y
tau_y
sigma_y
```

The sequence can be viewed as:

```text
Gaussian    A_x    tau_x    sigma_x    A_y    tau_y    sigma_y
   1         ...     ...       ...       ...     ...       ...
   2         ...     ...       ...       ...     ...       ...
   3         ...     ...       ...       ...     ...       ...
   .          .       .         .         .       .         .
  10         ...     ...       ...       ...     ...       ...
```

For `N` examples, the final arrays are:

```text
data_input.shape  = (N, 10, 6)
data_target.shape = (N, 18)
```

This is the reason for the post-processing step. It prepares the ordered Gaussian pulse representation for the recurrent model.

The builder also checks:

- whether the required MATLAB variables exist;
- the pulse tensor dimensions;
- the Gaussian centre ordering;
- the shared-width structure used by the simulator;
- NaN or infinite values;
- the 18-component target length;
- whether Pauli expectations are within the expected numerical range.

The processed dataset is saved as a compressed NumPy archive:

```text
data/processed/qubit_response_dataset.npz
```

More detail on the array ordering is given in:

```text
docs/data_schema.md
```

---

## 3. Train the black-box GRU

The recurrent model is defined in:

```text
src/qubit_surrogate/ml_model.py
```

Train the model with:

```bash
python python/train.py \
    --dataset data/processed/qubit_response_dataset.npz \
    --config config/default.json
```

### Model architecture

The black-box model follows the stacked GRU structure:

```text
Pulse sequence: (10, 6)
        |
        v
GRU, 60 units
return_sequences=True
        |
        v
GRU, 60 units
return_sequences=True
        |
        v
GRU, 18 units
activation=tanh
return_sequences=False
        |
        v
18 predicted measurement outcomes
```

In TensorFlow, the model is built from:

```python
GRU(units=60, return_sequences=True)
GRU(units=60, return_sequences=True)
GRU(units=18, activation="tanh", return_sequences=False)
```

The final `tanh` layer is used because the targets are Pauli expectation values and lie in the interval `[-1, 1]`.

The model is compiled with:

```text
Adam optimiser
learning rate = 0.012
mean squared error loss
```

The recurrent model is black-box. Its learning problem is simply:

```text
normalised pulse sequence -> final 18-component response
```

It is not given the equations used by the MATLAB simulator.

### Training outputs

Training files are saved in:

```text
results/models/
```

The main outputs are:

```text
best_model.keras
training_history.npz
split_indices.npz
run_config.json
```

The train, validation and test indices are saved so the same held-out examples can be used during evaluation.

---

## 4. Evaluate the trained model

After training, evaluate the held-out test examples:

```bash
python python/evaluate.py \
    --dataset data/processed/qubit_response_dataset.npz \
    --model results/models/best_model.keras \
    --split-file results/models/split_indices.npz
```

The evaluation code compares the GRU predictions with the MATLAB simulation targets.

The reported quantities include:

```text
global RMSE
global MAE
mean per-target R2
per-target RMSE
per-target R2
```

The evaluation also saves prediction data and figures.

Figures are written to:

```text
results/figures/
```

Model evaluation data are written to:

```text
results/models/
```

The intention is to test how accurately the recurrent black-box surrogate has learned the simulated pulse-response map on examples that were not used for training.

---

## 5. Use the trained model as a pulse controller

Once the GRU has been trained, its weights can be frozen and the model can be used as a differentiable response surrogate.

The controller code is in:

```text
src/qubit_surrogate/controller.py
```

and is connected to the trained model through:

```text
src/qubit_surrogate/ml_model.py
```

For example, to optimise a pulse for a Hadamard target:

```bash
python python/optimise_control.py \
    --model results/models/best_model.keras \
    --gate H \
    --epochs 2000
```

The predefined target gates are:

```text
I
X
Y
Z
H
S
T
```

### How does the controller work?

The trained GRU is frozen:

```text
trainable = False
```

The controller then generates a normalised Gaussian pulse sequence and passes it through the frozen GRU:

```text
trainable pulse parameters
        |
        v
frozen black-box GRU
        |
        v
predicted 18-component response
        |
        v
MSE against target response
```

The pulse family is kept consistent with the MATLAB simulation.

The controller optimises:

```text
10 x-channel amplitudes
10 y-channel amplitudes
1 shared Gaussian width
```

The Gaussian centres remain fixed at:

```text
k / (n_max + 1)
```

in the normalised representation.

The target response for a unitary `U` is calculated from:

```text
U rho U^dagger
```

for the same six initial states and the same three Pauli observables used in the simulation dataset.

### Controller outputs

The controller saves:

```text
normalised pulse sequence
MATLAB-style pulse_parameters
target response
GRU-predicted response
controller loss history
```

It also saves the pulse parameters as a MATLAB file so they can be loaded back into the simulator:

```text
gate_H_pulse_parameters.mat
```

The controller result is a surrogate prediction. The exported pulse should be run through the MATLAB simulator, or through an experiment, before treating it as an independently verified control solution.

---

## 6. Optional QuTiP example

A separate QuTiP example is included in:

```text
examples/qutip_coloured_dephasing_demo.py
```

Run:

```bash
python examples/qutip_coloured_dephasing_demo.py
```

This script demonstrates the same general stochastic-dephasing idea using QuTiP:

```text
generate coloured classical noise
        |
        v
evolve one qubit for each noise realisation
        |
        v
average the quantum states
        |
        v
calculate Pauli expectations and ensemble-state purity
```

This example is included as a complementary implementation of the physical mechanism. It is not used as an exact numerical reproduction of the MATLAB dataset generator.

---

## Reproducibility

Each MATLAB example is seeded using its example index:

```matlab
rng(idx_ex, 'twister')
```

The Python training workflow stores:

```text
random seed
training indices
validation indices
test indices
training history
run configuration
trained Keras model
```

This allows the same dataset split and model configuration to be inspected later.

---

## Tests

Run the Python tests with:

```bash
pytest
```

You can also check that all Python source files compile:

```bash
python -m compileall src python tests
```

GitHub Actions runs the test suite on pushes to `main` and on pull requests.

The workflow file is:

```text
.github/workflows/ci.yml
```

---

## Adding changes

Before committing changes, run:

```bash
pytest
python -m compileall src python tests
```

Then add and commit the files:

```bash
git add .
git commit -m "Describe the change"
git push
```

For larger changes, use a separate branch:

```bash
git checkout -b feature/my-change
```

After making and testing the changes:

```bash
git add .
git commit -m "Describe the change"
git push -u origin feature/my-change
```

A pull request can then be opened on GitHub.

---

## Additional documentation

More detailed notes are available in:

```text
docs/data_schema.md
docs/model.md
docs/controller.md
matlab/README.md
```

---


See:

```text
THIRD_PARTY_NOTICES.md
``` 
