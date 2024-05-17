# On PI controllers for updating Lagrange multipliers in constrained optimization

## About

Official implementation for the paper [On PI controllers for updating Lagrange multipliers in constrained optimization](https://openreview.net/forum?id=1khG2xf1yt).
This codebase implements the $\nu \hspace{-0.5ex}$ PI optimizer. Relative to gradient ascent, $\nu \hspace{-0.5ex}$ PI improves the training dynamics of Lagrange multipliers in constrained optimization problems.

We use the [Cooper library](https://github.com/cooper-org/cooper) for implementing and solving constrained deep-learning optimization problems.

## Usage

### Dependencies
We use Python 3.10.10.

To install necessary dependencies, run the following command:
```bash
pip install -r requirements.txt
```

### WandB
We use [Weights and Biases](https://wandb.ai/) to track our experiments. To use WandB, you need to create an account and login.
Then setup a new project.

To make authentication easier, we recommend setting up an environment variable for your API key.
You can find your API key in your account settings. Then add the following line to your .bashrc or .zshrc file:
```bash
export WANDB_API_KEY=<your_api_key>
```

### Required enviroment variables

We use [`dotenv`](https://github.com/theskumar/python-dotenv) to manage environment variables. Please create a `.env` file in the root directory of the project and add the following variables:

```
# Location of the directory containing your datasets
DATA_DIR=

# The directory where the results will be saved
CHECKPOINT_DIR=

# If you want to use Weights & Biases, add entity and project name here
WANDB_ENTITY=
WANDB_PROJECT=

# Directory for Weights & Biases local storage
WANDB_DIR=

# Directory for logs created by submitit
SUBMITIT_DIR=

# Directory for saving large datasets
SLURM_TMPDIR=
```

### Submitit

We use [submitit](https://github.com/facebookincubator/submitit) to facilitate experiments on SLURM clusters. For local execution, you can use:

```bash
    --resources_config=configs/resources.py:"cluster=debug"
```
Alternatively,
```bash
    --resources_config=configs/resources.py:"cluster=local"
```

| Cluster | Is local? | Is SLURM? | Is interactive? | Is background? | Notes |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `debug` | ✅ | 🚫 | ✅ | 🚫 | For local execution on your machine or on a compute node |
| `local` | ✅ | 🚫 | 🚫 | ✅ | This could be locally on your machine or on a compute node |
| `unkillable` | 🚫 | ✅ | 🚫 | ✅ | To run jobs on SLURM `unkillable` partition |
| `main` | 🚫 | ✅ | 🚫 | ✅ | To run jobs on SLURM `main` partition |
| `long` | 🚫 | ✅ | 🚫 | ✅ | To run jobs on SLURM `long` partition |


Examples of valid resource configurations:
- `"cluster=debug"` # This can be used to debug your code locally or on a compute node
- `"cluster=local tasks_per_node=1 use_ddp=False"` # This would only use 1 GPU even if more are available
- `"cluster=local tasks_per_node=2 use_ddp=True"`
- `"cluster=unkillable"`
- `"cluster=main tasks_per_node=2 use_ddp=True"`
- `"cluster=long tasks_per_node=2 use_ddp=True"`


### Running experiments

A hard margin SVM experiment on a two-class version of the Iris dataset can be run using the following command:
```bash
python main.py \
    --model_config=configs/model.py:"name=LinearModel" \
    --data_config=configs/data.py:"dataset_name=iris" \
    --optim_config=configs/optim.py:"cooper_optimizer=AlternatingDualPrimalOptimizer primal_optimizer=sgd dual_optimizer=sgd dual_optimizer.kwargs.lr=0.01" \
    --task_config=configs/task.py:"task=max_margin" \
    --metrics_config=configs/metrics.py:"metrics=classification" \
    --resources_config=configs/resources.py:"cluster=debug tasks_per_node=1 use_ddp=False" \
    --config.train.total_epochs=100 \
    --config.logging.wandb_mode=disabled
```

A sparsity constrained classification experiment on CIFAR10 can be run using the following command:
```bash
python main.py \
    --model_config=configs/model.py:"name=CifarSparseResNet18" \
    --data_config=configs/data.py:"dataset_name=cifar10" \
    --optim_config=configs/optim.py:"cooper_optimizer=AlternatingDualPrimalOptimizer primal_optimizer=sgd dual_optimizer=sgd dual_optimizer.kwargs.lr=0.01" \
    --task_config=configs/task.py:"task=sparsity cmp_kwargs.target_sparsities=(0.3,)" \
    --metrics_config=configs/metrics.py:"metrics=classification" \
    --resources_config=configs/resources.py:"cluster=debug tasks_per_node=1 use_ddp=False" \
    --config.train.total_epochs=100 \
    --config.logging.wandb_mode=disabled
```

A fairness constrained experiment on the Adult dataset can be run using the following command:
```bash
python main.py \
    --model_config=configs/model.py:"name=MLP" \
    --data_config=configs/data.py:"dataset_name=adult" \
    --optim_config=configs/optim.py:"cooper_optimizer=AlternatingDualPrimalOptimizer primal_optimizer=sgd dual_optimizer=sgd dual_optimizer.kwargs.lr=0.01" \
    --task_config=configs/task.py:"task=fairness" \
    --metrics_config=configs/metrics.py:"metrics=group_classification" \
    --resources_config=configs/resources.py:"cluster=debug tasks_per_node=1 use_ddp=False" \
    --config.train.total_epochs=100 \
    --config.logging.wandb_mode=disabled
```

## Citing this work

To cite this work, please use the following BibTeX entry:

```bibtex
@inproceedings{sohrabi2024nuPI,
  title={{On PI controllers for updating Lagrange multipliers in constrained optimization}},
  author={Sohrabi, Motahareh and Ramirez, Juan and Zhang, Tianyue H. and Lacoste-Julien, Simon and Gallego-Posada, Jose},
  booktitle={ICML},
  year={2024}
}
```
