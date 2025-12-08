package main

import (
	"context"
	"dagger-pipeline/internal/dagger"
)

type DaggerPipeline struct{}

// Train runs the training pipeline: install dependencies, pull data, process features, and train the model.
// It returns the models directory containing the trained artifacts.
func (m *DaggerPipeline) Train(
	ctx context.Context,
	// The source directory containing the project code
	source *dagger.Directory,
) *dagger.Directory {
	// Start with a Python container
	container := dag.Container().
		From("python:3.10").
		WithDirectory("/workspace", source).
		WithWorkdir("/workspace").
		// Install dependencies
		WithExec([]string{"pip", "install", "-r", "requirements.txt"}).
		// Pull data using DVC, fallback to repro (download) if pull fails
		WithExec([]string{"sh", "-c", "dvc pull || dvc repro --force data/raw/raw_data.csv.dvc"}).
		// Run pipeline steps
		WithExec([]string{"python", "mlops_src/templ_dataset.py"}).
		WithExec([]string{"python", "mlops_src/templ_features.py"}).
		WithExec([]string{"python", "mlops_src/modeling/templ_train.py"})

	// Return the models directory
	return container.Directory("/workspace/models")
}

// Test runs the model inference test.
func (m *DaggerPipeline) Test(
	ctx context.Context,
	// The source directory containing the project code
	source *dagger.Directory,
) *dagger.Container {
	return dag.Container().
		From("python:3.10").
		WithDirectory("/workspace", source).
		WithWorkdir("/workspace").
		WithExec([]string{"pip", "install", "-r", "requirements.txt"}).
		// We assume data is needed for prediction too, or at least the model
		WithExec([]string{"sh", "-c", "dvc pull || dvc repro --force data/raw/raw_data.csv.dvc"}).
		WithExec([]string{"python", "mlops_src/modeling/templ_predict.py"})
}
