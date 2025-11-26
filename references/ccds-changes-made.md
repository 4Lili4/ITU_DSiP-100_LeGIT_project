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
