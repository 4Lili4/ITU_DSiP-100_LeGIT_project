package main

import (
	"context"
	"log"

	"dagger.io/dagger"
)

func runPython(ctx context.Context, client *dagger.Client, src *dagger.Directory) *dagger.Container {
	return client.Container().
		From("python:3.10").
		WithDirectory("/workspace", src).
		WithWorkdir("/workspace").
		WithExec([]string{"pip", "install", "-r", "requirements.txt"})
}

func main() {
	ctx := context.Background()

	client, err := dagger.Connect(ctx)
	if err != nil {
		log.Fatalf("Failed to connect to Dagger: %v", err)
	}
	defer client.Close()

	// Load entire repo from project root
	project := client.Host().Directory("..")

	//set up base container:
	pipeline := runPython(ctx, client, project)

	// Data prep and cleaining
	log.Println("Running I_data_prep.py and II_data_cleaning.py...")
	pipeline = pipeline.WithExec([]string{
		"python", "mlops_src/dataset-py/I_data_prep.py",
	}).WithExec([]string{
		"python", "mlops_src/dataset-py/II_data_cleaning.py",
	})

	// Data splitting
	log.Println("Running III_data_splitting.py...")
	pipeline = pipeline.WithExec([]string{
		"python", "mlops_src/features-py/III_data_splitting.py",
	})

	// Model training and selection
	log.Println("Running IV_model_training.py …")
	pipeline = pipeline.WithExec([]string{
		"python", "mlops_src/modeling/train-py/IV_model_training.py",
	})

	log.Println("Running V_model_test.py …")
	pipeline = pipeline.WithExec([]string{
		"python", "mlops_src/modeling/train-py/V_model_test.py",
	})

	log.Println("Running VI_model_selection.py …")
	finalStep := pipeline.WithExec([]string{
		"python", "mlops_src/modeling/train-py/VI_model_selection.py",
	})

	// Force evaluation to ensure the script runs and the file exists
	if _, err := finalStep.Sync(ctx); err != nil {
		log.Fatalf("ML pipeline execution failed at the final step: %v", err)
	}

	// Saving model artifacts
	log.Println("Exporting model artifact …")

	modelFile := finalStep.File("models/lead_model_lr.pkl")

	_, err = modelFile.Export(ctx, "../models/model.pkl")
	if err != nil {
		log.Fatalf("Failed to export trained model: %v", err)
	}
	log.Println("Export complete")

}
