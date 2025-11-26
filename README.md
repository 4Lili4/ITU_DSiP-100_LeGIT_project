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

The full data structure can be found under `references/data-structure.md`. 

Files that were modified from the original given structure can be found in `references/ccds-changes-made.md`.

## TODO:
- migrate the `references/Notes from last lecture.docx` file from references out of the repo
- sort out the `notebooks/old-to-be-merged` folder