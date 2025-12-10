package main

import (
	"context"
	"dagger-pipeline/internal/dagger"
)

type DaggerPipeline struct{}

// Test runs the pytests
func (m *DaggerPipeline) Test(
	ctx context.Context,
	source *dagger.Directory,
) *dagger.Container {
	return dag.Container().
		From("python:3.10").
		WithEnvVariable("PYTHONPATH", "/workspace").
		WithDirectory("/workspace", source).
		WithWorkdir("/workspace").
		WithExec([]string{"pip", "install", "-r", "requirements.txt"}).
		// Pull data if relevant for tests
		WithExec([]string{"sh", "-c", "dvc pull || dvc update data/raw/raw_data.csv.dvc"}).
		WithExec([]string{"python", "mlops_src/modeling/test_training.py"}).
		WithExec([]string{"python", "mlops_src/modeling/test_inference.py"})
}

// Train runs the training pipeline: install dependencies, pull data, process features, and train the model.
// It returns the models directory containing the trained artifacts.
func (m *DaggerPipeline) Train(
	ctx context.Context,
	// When called from github flow '--source .' is passed into the function which mounts the entire repo
	source *dagger.Directory,
) *dagger.Directory {
	container := dag.Container().
		From("python:3.10").
		WithEnvVariable("PYTHONPATH", "/workspace"). //setup container
		WithDirectory("/workspace", source).
		WithWorkdir("/workspace").
		// Install dependencies
		WithExec([]string{"pip", "install", "-r", "requirements.txt"}).
		// Pull data using DVC, fallback to update (fetch from url) if pull fails
		WithExec([]string{"sh", "-c", "dvc pull || dvc update data/raw/raw_data.csv.dvc"}).
		// Run pipeline steps
		WithExec([]string{"python", "mlops_src/templ_dataset.py"}).
		WithExec([]string{"python", "mlops_src/templ_features.py"}).
		WithExec([]string{"python", "mlops_src/modeling/templ_train.py"})

	// Return the models directory (all artifacts)
	return container.Directory("/workspace/models")
}

// Predict runs the model inference
func (m *DaggerPipeline) Predict(
	ctx context.Context,
	source *dagger.Directory,
) *dagger.Container {
	return dag.Container().
		From("python:3.10").
		WithEnvVariable("PYTHONPATH", "/workspace").
		WithDirectory("/workspace", source). //same setup
		WithWorkdir("/workspace").
		WithExec([]string{"pip", "install", "-r", "requirements.txt"}).
		// We assume data is needed for prediction too, or at least the model
		WithExec([]string{"sh", "-c", "dvc pull || dvc update data/raw/raw_data.csv.dvc"}).
		WithExec([]string{"python", "mlops_src/modeling/templ_predict.py"})
}