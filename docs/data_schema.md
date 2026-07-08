# Data schema

## Raw MATLAB file

Each `dataset_XXXX.mat` file contains two variables.

### `pulse_parameters`

Expected shape:

```text
(2, 10, 3)
```

Axis meanings:

| Axis | Meaning |
|---|---|
| 0 | control channel: x, y |
| 1 | ordered Gaussian component |
| 2 | amplitude, centre, width |

The public simulation uses fixed equally spaced centres and one width shared across all Gaussian components and both channels for a given example.

### `data`

Expected flattened length:

```text
18
```

Initial-state order:

```text
+X, -X, +Y, -Y, +Z, -Z
```

Observable order for each state:

```text
<X>, <Y>, <Z>
```

Therefore the flattened target order is:

```text
+X/<X>, +X/<Y>, +X/<Z>,
-X/<X>, -X/<Y>, -X/<Z>,
+Y/<X>, +Y/<Y>, +Y/<Z>,
-Y/<X>, -Y/<Y>, -Y/<Z>,
+Z/<X>, +Z/<Y>, +Z/<Z>,
-Z/<X>, -Z/<Y>, -Z/<Z>
```

## Processed archive

`build_dataset.py` writes:

```text
data_input
data_target
source_files
feature_names
target_names
```

### `data_input`

Shape:

```text
(N, 10, 6)
```

Feature order:

```text
A_x, tau_x, sigma_x, A_y, tau_y, sigma_y
```

The features are mapped to the public pulse-domain normalisation:

- amplitudes: `[−A_max(sigma), +A_max(sigma)] -> [0, 1]`
- centres: `tau / T`
- widths: `[sigma_min, sigma_max] -> [0, 1]`

### `data_target`

Shape:

```text
(N, 18)
```

Targets are final ensemble-averaged Pauli expectation values.
