# CumuTopoNet

CumuTopoNet is a Python package for drone radio-frequency classification using higher-order cumulants, temporal features, and topological data analysis. It includes data preparation, model training, evaluation, reporting, and diagnostic analysis commands.

## Requirements

- Python 3.11
- Sufficient local storage for the prepared dataset
- An NVIDIA CUDA GPU for full training runs

The test suite and synthetic smoke workflow can run on CPU.

## Installation

```bash
bash scripts/install.sh
source .venv/bin/activate
python -m pytest -q
cumutopo --help
```

The installer creates an isolated environment and installs the pinned dependencies from `requirements-core.lock`.

## Dataset

Download DroneDetect V2 from its [IEEE DataPort page](https://ieee-dataport.org/open-access/dronedetect-dataset-v2). The project accepts either the original ZIP archive or an extracted directory of `.dat` recordings.

Place the dataset at `data/DroneDetect_V2.zip`, or set another path in a local configuration file. Paths in YAML files are resolved relative to the configuration file.

```bash
cp configs/matched-128.example.yaml configs/local-matched.yaml
cp configs/full-1024.example.yaml configs/local-full.yaml
```

See [the dataset format](docs/DATA_FORMAT.md) for directory naming, sample encoding, and optional session metadata.

## Matched 128-sample workflow

Prepare the local environment and data:

```bash
cumutopo matched-128 inventory -c configs/local-matched.yaml
cumutopo matched-128 doctor -c configs/local-matched.yaml --tests all
cumutopo matched-128 doctor -c configs/local-matched.yaml --cuda --device cuda:0 --worker-name local
cumutopo matched-128 prepare -c configs/local-matched.yaml --benchmark
cumutopo matched-128 worker -c configs/local-matched.yaml --name local --device cuda:0 --benchmark
cumutopo matched-128 plan -c configs/local-matched.yaml --budget
cumutopo matched-128 prepare -c configs/local-matched.yaml
cumutopo matched-128 plan -c configs/local-matched.yaml --pilots
```

Start the local training process:

```bash
cumutopo matched-128 worker -c configs/local-matched.yaml --name local --device cuda:0
```

After the learning-rate pilots finish, advance the workflow and inspect its state:

```bash
cumutopo matched-128 status -c configs/local-matched.yaml --advance
cumutopo matched-128 status -c configs/local-matched.yaml
```

When training is complete, evaluate and generate reports:

```bash
cumutopo matched-128 evaluate -c configs/local-matched.yaml --device cuda:0
cumutopo matched-128 report -c configs/local-matched.yaml
```

Use `--output-dir` with `report` to write the report somewhere other than the configured workspace.

## Full 1,024-sample workflow

```bash
cumutopo full-1024 inventory -c configs/local-full.yaml
cumutopo full-1024 prepare -c configs/local-full.yaml
cumutopo full-1024 prepare-sensitivity -c configs/local-full.yaml
cumutopo full-1024 plan -c configs/local-full.yaml --suite standard-1024
cumutopo full-1024 worker -c configs/local-full.yaml --name local --device cuda:0
cumutopo full-1024 status -c configs/local-full.yaml
cumutopo full-1024 report -c configs/local-full.yaml
```

The extended workflow is enabled with `--suite extended-1024`. Its LSTM learning-rate choice is finalized with `status --advance-lstm` after the pilot runs finish.

## Analysis commands

```bash
cumutopo analyze interference -c configs/local-full.yaml
cumutopo analyze subgroups -c configs/local-full.yaml
cumutopo analyze cost -c configs/local-full.yaml --device cuda:0
```

Generated files are written beneath the configured workspace. Dataset files, prepared caches, checkpoints, plots, and result files are intentionally excluded from the repository.

## Project documentation

- [Method and model details](docs/METHOD.md)
- [Local workflows](docs/WORKFLOWS.md)
- [Dataset format](docs/DATA_FORMAT.md)
- [Third-party components](THIRD_PARTY.md)
