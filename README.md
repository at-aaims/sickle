Below is the updated README script incorporating your preferred usage examples that use YAML configuration files along with optional command-line overrides:

```markdown
# SICKLE

**SICKLE** (Sparse Intelligent Curation frameworK for Learning Efficiency) is a tool designed to extract data with the highest probabilistic information content, thereby reducing the cost of training large models.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Examples](#examples)
- [Advanced Topics](#advanced-topics)
- [License](#license)
- [Additional Resources](#additional-resources)

## Overview

SICKLE helps "separate the wheat from the chaff" by using various subsampling methods (e.g., maxent, random) to extract the most informative data segments. It supports both training and testing modes and can be run on different systems (e.g., local laptop, Frontier).

## Installation

1. **Environment Setup:**  
   Activate the required Python virtual environments:
   ```bash
   source /path/to/venv/bin/activate
   ```
2. **Dependencies:**  
   Ensure required modules (e.g., `cray-python`, `rocm`) are loaded:
   ```bash
   module load cray-python/3.10.10 rocm
   ```
3. **Download/Clone Repository:**  
   Clone the repository to your local machine.

## Usage

SICKLE is run from the command line. It supports both direct command-line specification of parameters and YAML configuration files. When a YAML configuration file is provided, its settings are used as defaults; any additional command-line switches will override the YAML values.

### Subsampling

Instead of specifying every parameter on the command line, you can use a YAML configuration file. For example:

- **Using the YAML file only:**
  ```bash
  python subsample.py config/OF/default.yaml
  ```

- **Overriding specific parameters:**
  ```bash
  python subsample.py config/OF/default.yaml --plot
  ```

### Training

For training, a similar approach is used. For example:

- **Using the YAML file only:**
  ```bash
  python -u train.py config/OF/default.yaml
  ```

- **Overriding specific parameters (e.g., epochs):**
  ```bash
  python -u train.py config/OF/default.yaml --epochs 3000
  ```

For a complete list of options, see the [`args.py`](./args.py) file.

## Configuration

SICKLE uses YAML configuration files to set parameters. All configurations are flattened, which means they don't need to be nested under a hierarchy. Below is an example configuration snippet:

```yaml
shared:
  dims: 3
  dtype: sst-binary
  noseed: true
  input_vars: [u, v, w, r]
  output_vars: [p, pv]
  cluster_var: [p, pv]
  nx: 514
  ny: 512
  nz: 256
  gravity: z
  fileprefix: "SST-P1-H{hypercubes}-cubes{num_hypercubes}-X{method}-ns{num_samples}-window{window}"

subsample:
  hypercubes: maxent
  num_hypercubes: 32
  method: maxent  # or random
  path: /path/to/data/
  num_samples: 3277
  num_clusters: 20
  nxsl: 32
  nysl: 32
  nzsl: 32

train:
  epochs: 1000
  batch: 16
  target: p_full
  window: 1
  arch: MLP_transformer
  sequence: true
```

*Note:* Adjust the YAML details as needed for your use cases.

## Examples

Detailed examples (including commands for testing on laptops, Frontier, parallel runs, and flow over cylinder cases) are provided in a separate file: [EXAMPLES.md](./EXAMPLES.md).

## Advanced Topics

- **Parallel Processing:**  
  SICKLE supports parallel execution (e.g., using `srun` for MPI-based tests).
- **Mixed Precision and Scalability:**  
  Options such as mixed precision (`amp`) and network architectures like `MLP_transformer` are available.
- **Integration with PyTorch:**  
  See the [PyTorch Frontier Documentation](https://docs.olcf.ornl.gov/software/python/pytorch_frontier.html) for further details on the training environment.

## License

This project is licensed under the **MIT License**.

For more details, see the [LICENSE](./LICENSE) file.

## Additional Resources

- [PyTorch Frontier Documentation](https://docs.olcf.ornl.gov/software/python/pytorch_frontier.html)
- For further details on configuration and command-line options, refer to the inline comments in [`args.py`](./args.py).
```
