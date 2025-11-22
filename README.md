# MLOps Project 2025 - group 100% LeGIT

## Overview

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Refactored MLOps project for the ITU course 'Data Science in Production: MLOps and Software Engineering' (Autumn 2025), set up by the course group '100% LeGIT'. 

The above sentence is also used as the description of the CCDS template

Project structure was set up using the Cookiecutter Data Science (CCDS) template (v2) with the parameters listed below alongside the reasoning for it. The resulting data structure can be viewed below the parameters table.

## Parameters & Reasoning

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

```
├── .dvc                <- DVC directory for DVC specific files (things like running the dagger workflow would go here)
│   │
│   ├── .gitignore          <- gitignore specifically for temporary files made by DVC
│   │
│   └── config              <- DVC config file (defines remote cloud storage for data files to pull from)
│
│
├── .github             <- GitHub directory for GitHub specific files (things like running the dagger workflow would go here)
│   │
│   └── workflows           <- Directory for GitHub workflows
│       │
│       └── test_action.yml     <- Test action workflow file (not sure what for yet)
│
│
├── dagger              <- Dagger directory for dagger workflow files (kept as a separate directory for local pipeline execution)
│   │
│   └──
│
│
├── data                <- Storage for data, organized by processing stage
│   │
│   ├── external            <- Data from third party sources (i.e. pre-made test data)
│   │   │
│   │   ├── X_test.csv          <- Test features dataset
│   │   │
│   │   └── y_test.csv          <- Test target dataset
│   │
│   ├── interim             <- Intermediate data that has been transformed
│   │   │
│   │   └──
│   │
│   ├── processed           <- The final data sets for modeling
│   │   │
│   │   └──
│   │
│   └── raw                 <- The original, immutable data dump
│       │
│       └── raw_data.csv.dvc    <- DVC file pointer and verification for the raw data
│
│
├── docs                <- A default mkdocs setup from mkdocs.org for documentation and documentation generation
│   │
│   ├── docs                <- The default directory for mkdocs documentation
│   │   │       
│   │   ├── getting-started.md  <- 
│   │   │       
│   │   └── index.md            <- 
│   │
│   ├── mkdocs.yml          <- The configuration file for the mkdocs documentation
│   │
│   └── README.md           <- Readme for mkdocs
│
│
├── mlops_src           <- Source code for use in this project
│   │
│   ├── modeling            <- Directory for model training and inference code
│   │   │  
│   │   ├── __init__.py         <- Makes the directory (modeling) a Python package
│   │   │
│   │   ├── predict.py          <- Code to run model inference with trained models (e.g. API endpoint)
│   │   │         
│   │   └── train.py            <- Code to train models (e.g. training loop)
│   │
│   ├── __init__.py         <- Makes the directory (mlops_src) a Python module
│   │
│   ├── config.py           <- Store useful variables and configuration (e.g. paths to data, model parameters, etc.)
│   │
│   ├── dataset.py          <- Scripts to download or generate data (e.g. data loading, data cleaning, data preprocessing)
│   │
│   ├── features.py         <- Code to create features for modeling (e.g. feature engineering, feature selection)
│   │
│   └── plots.py            <- Code to create visualizations (e.g. data visualizations, model visualizations)
│
│
├── models              <- Trained and serialized models, model predictions, or model summaries
│   │   
│   └── lead_model_lr.pkl   <- Model artifact (pickle file) for the lead prediction model
│
│
├── notebooks           <- Jupyter notebooks. Naming convention is a number (for ordering), the creator's initials
│   │                      and a short `-` delimited description, e.g. `1.0-jqp-initial-data-exploration.ipynb`
│   └── main.ipynb          <- The old main (huge) notebook for the project
│
│
├── references          <- Data dictionaries, manuals, and all other explanatory materials
│   │
│   ├── diagrams.excalidraw      <- Diagram that can be visualized in excalidraw
│   │
│   ├── project-architecture.png <- A .png to be loaded in project-description.md of the architecture diagram
│   │       
│   └── project-description.md   <- Old project description file
│
│
├── reports             <- Generated analysis as HTML, PDF, LaTeX, etc.
│   │   
│   └── figures             <- Generated graphics and figures to be used in reporting
│       │ 
│       └──
│
│
├── tests               <- Contains automated tests (for future use, using pytest)
│   │
│   └── test_data.py        <- Example test script
│   
│
├── .dvcignore          <- DVC ignore file to ignore specific files or directories
│
├── .env                <- Environment variables
│
├── .gitignore          <- Git ignore file to ignore specific files or directories
│
├── LICENSE             <- MIT License file
│
├── Makefile            <- Makefile with convenience commands like `make data` or `make train`
│
├── pyproject.toml      <- Project configuration file with package metadata for mlops_src and configuration for tools
│
├── README.md           <- The top-level README for viewers of this project
│
└── requirements.txt    <- The requirements file for reproducing the environment
```

## Things that were moved, created, updated or ignored from the original version

Old project directory is referred to as `mlops_old`.

Moved from `mlops_old` to `ITU_DSiP-100_LeGIT_project`:
- `.github` to `.github` (contains github specific actions so it has to be in source)
- `docs/project-architecture.png` to `references/project-architecture.png`
- `docs/diagrams.excalidraw` to `references/diagrams.excalidraw`
- `notebooks/.dvc` to `.dvc` (this way it has an overview of all the data directories)
- `notebooks/artifacts/lead_model_lr.pkl` to `models/lead_model_lr.pkl` (its the model artifact so we want to have it in the actual models folder)
- `notebooks/artifacts/raw_data.csv.dvc` to `data/raw/raw_data.csv.dvc` (its where we get our raw data from, using dvc, so it has to be at the raw data directory)
- `notebooks/artifacts/X_test.csv` to `data/external/X_test.csv` (contains test data features for testing purposes)
- `notebooks/artifacts/y_test.csv` to `data/external/y_test.csv` (contains test data target for testing purposes)
- `notebooks/.dvcignore` to `.dvcignore` (a global ignore for DVC related files)
- `notebooks/main.ipynb` to `notebooks/main.ipynb` (its the main notebook so we want to keep it in notebooks)
- `README.md` to `references/project-description.md` (its the project description so it had to be renamed)

Created in `ITU_DSiP-100_LeGIT_project`:
- `mlops_old/dagger` (contains the dagger workflow files)

Updated in `ITU_DSiP-100_LeGIT_project`:
- `.gitignore` to include data/raw/raw_data.csv in the 'Data' group
- `requirements.txt` to include the old requirements
- `mlops_src/modeling/predict.py` to include the old code from model_inference.py

Ignored in `mlops_old`:
- `action.yml` (it's empty, so for now we ignore it. It's for github actions and we will need to make a new one anyways somewhere when we get there)

## TODO:
- migrate the `references/Notes from last lecture.docx` file from references out of the repo
- sort out the `notebooks/old-to-be-merged` folder