# MLOps Project 2025 - group `100% LeGIT`


## Overview

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Refactored MLOps project for the ITU course 'Data Science in Production: MLOps and Software Engineering' (Autumn 2025), set up by the course group '100% LeGIT'. 

The above sentence is also used as the description of the CCDS template

Project structure was set up using the Cookiecutter Data Science (CCDS) template (v2) with the parameters listed below alongside the reasoning for it. The resulting data structure can be viewed below the parameters table.


## Running the code

There are two ways to run the workflow. Triggering it on the cloud, or locally running it.

### On the cloud

The `train` workflow is triggered either on a Pull Request to `main`, on changes to `main`, or manually. It runs both the `train`, `test`, `predict` pipelines, and the `validator` provided within the MLOps course.

The `test` and `predict` workflows can be separately triggered manually. To manually triger a workflow, head over to `Actions` tab in GitHub, select the workflow you want to trigger, select branch `main`, and finally press `Run Workflow`. Afterwards the workflow will appear upon a refresh of the page, and can be inspected while running or post-run. 

### Running Locally with Dagger

This project uses [Dagger](https://dagger.io/) to define and run its CI/CD pipeline. You can execute the same pipelines locally that run in GitHub Actions.

### Prerequisites

Ensure you have the following installed:

1.  **Docker** (or a compatible container runtime) - Dagger runs steps in containers.
2.  **Dagger CLI** (v0.19.8 or later recommended).
3.  **Go** (v1.24 or later) - The Dagger pipeline is written in Go.
4.  **Python** (v3.10) - Required for local development outside Dagger.

> **Note:** The `dagger` module is located in the `dagger/` directory. All commands below assume you are running from the **root** of the repository.

### Running Pipeline Steps

You can run individual pipeline steps (functions) using the `dagger call` command.

#### 1. Run Tests

To run the `pytest` suite inside the Dagger container:

```bash
dagger call -m dagger test --source .
```

This will:
- Spin up a container.
- Install dependencies from `requirements.txt`.
- Pull/Update DVC data.
- Run `pytest tests/`.

#### 2. Run Training Pipeline

To run the full training pipeline (processing -> training -> evaluation):

```bash
dagger call -m dagger train --source . export --path ./models
```

This will run the training steps and export the resulting artifacts (saved models) to your local `models/` directory.

#### 3. Run Prediction

To run the prediction/inference step:

```bash
dagger call -m dagger predict --source . sync
```

This executes the `mlops_src/modeling/templ_predict.py` script within the container. We use `sync` to ensure the container execution completes.


## CCDS Parameters & Reasoning

| Parameter | Value | Reasoning |
| :--- | :--- | :--- |
| `project_name` | `ITU_DSiP-100_LeGIT_project` | No specific request for name so we use a self-picked name. |
| `repo_name` | `ITU_DSiP-100_LeGIT_project` | Kept consistent with project name for simplicity and for ease of applying the template to this project. |
| `module_name` | `mlops_src` | By default this takes repo_name, but we use a slight variation of the generic "src" name to keep consistent with its intended use as the source code package/module. |
| `author_name` | `100% LeGIT` | Our group name in the MLOps course. |
| `description` | `*as stated above` | Describes the purpose of this project. |
| `python_version_number` | `3.10` | Selected 3.10 for maximum compatibility. While 3.12 is newer, some data science libraries may not be fully up to date. 3.10 is a stable, widely supported baseline. |
| `dataset_storage` | `none` | We are meant to work on this on local storage so we do not need to use a cloud storage. |
| `environment_manager` | `virtualenv` | Chosen for its lightweight nature and standard usage in CI/CD. While Conda is powerful, it can be slower and heavier; virtualenv is sufficient for this project's scope. |
| `dependency_file` | `requirements.txt` | Simple dependency list, standard format compatible with most tools and easy to read. |
| `pydata_packages` | `basic` | This is a DS/MLOps project, so core libraries (pandas, numpy) are guaranteed requirements. Pre-installing them saves manual setup time. |
| `testing_framework` | `pytest` | The modern industry standard. It offers a simpler syntax and an easier learning curve compared to unittest. |
| `linting_and_formatting` | `ruff` | Standard modern linter/formatter. |
| `open_source_license` | `MIT` | This will likely be available in the future publicly so we use MIT license, as it is permissive and allows for easy contribution. |
| `docs` | `mkdocs` | Chosen for its simplicity and ease of use for documentation generation. |
| `include_code_scaffold` | `Yes` | Provides a reference architecture (e.g., where to put data processing vs. training code) to guide the migration from the notebook. |


## Generated Data Structure

The full data structure can be found under `references/data-structure.md`. 

Files that were modified from the original given structure can be found in `references/ccds-changes-made.md`.

### Summary of each sub-directory:

<details>
<summary><strong>.dvc/</strong></summary>

DVC directory for DVC specific files, enables versioning of large files that should not be stored directly in Git.

- Data files in `data/raw/` and other large artifacts are tracked via `.dvc` files
- Ensure reproducibility across experiments and pipeline runs

Files: 
- `.gitignore` - gitignore specifically for temporary files made by DVC
- `config` - DVC config file (defines remote cloud storage for data files to pull from)

</details>

<details>
<summary><strong>.github/workflows/</strong></summary>

This directory defines the **GitHub Actions workflows** that trigger and execute the model creation, testing, and validation pipeline automatically by triggering dagger.

</details>

<details>
<summary><strong>dagger/</strong></summary>

Dagger directory for dagger workflow files (kept as a separate directory for local pipeline execution)

Contains functions to run the full ML pipeline, pytests, and artitfact validation

Key files:
- `main.go` – Entry point for the Dagger pipeline
- `dagger.json` – Dagger configuration
- `go.mod`, `go.sum` – Go dependencies

</details>

<details>
<summary><strong>data/</strong></summary>

Storage for data, organized by processing stage, following cookiecutter data science principles:

- `raw/` – Immutable raw data (tracked with DVC)
- `external/` – Data from third party sources (i.e. pre-made test data)
- `interim/` – Intermediate outputs that has been transformed
- `processed/` – Final datasets for modelling
- `raw` - The original, immutable data dump

</details>

<details>
<summary><strong>docs/</strong></summary>

A default mkdocs setup from mkdocs.org for documentation and documentation generation, including:
- `mkdocs.yml` - The configuration file for the mkdocs documentation

</details>

<details>
<summary><strong>mlops_src/</strong></summary>

Source code for use in this project.
All logic here is written as reusable, modular templates that can be executed both locally and in CI.

Structure:
- `templ_dataset.py` – Preprocessing
- `templ_features.py` – Feature selection
- `modeling/`
  - `templ_train.py` – Model training
  - `templ_predict.py` – Inference logic
- `templ_config.py` – Reusable configuration and useful variables
- `__init__.py` – Makes the directory (mlops_src) a Python module
- 
</details>

<details>
<summary><strong>models/</strong></summary>

Stores serialized model outputs and artifacts, including:
- Trained model files (`.pkl`, `.json`)
- Evaluation summaries and performance metrics

The final artifact will be validated automatically through github actions.

</details>

<details>
<summary><strong>notebooks/</strong></summary>

Jupyter notebooks. Naming convention is a number (for ordering), the creator's initials and a short `-` delimited description, e.g. `1.0-jqp-initial-data-exploration.ipynb`

Once logic is validated in notebooks, it is **refactored into `mlops_src/` templates** to ensure reproducibility and CI compatibility.

</details>

<details>
<summary><strong>references/</strong></summary>

Contains supplementary explanatory materials for context through the development stages of this project:
- `project-architecture.png` - A .png to be loaded in project-description.md of the architecture diagram
- `diagrams.excalidraw` - Diagram that can be visualized in excalidraw
- Project description

</details>

<details>
<summary><strong>reports/</strong></summary>

Generated analysis as HTML, PDF, LaTeX, etc.
For future expansion of a typical project, it would store figures and visual outputs generated during analysis and evaluation.

</details>

<details>
<summary><strong>tests/</strong></summary>

Contains automated tests using pytest for:
- Model training pipelines
- Inference and prediction logic

Tests are executed in CI to ensure correctness and stability before merging or deployment.

</details>
