package main

import (
	"context"
	"log"

	"dagger.io/dagger"
)

func runPython(ctx context.Context, client *dagger.Client, src *dagger.Directory) *dagger.Container {
	return client.Container().
		From("python:3.10").
		WithDirectory("/workspace", src). //setup container
		WithWorkdir("/workspace").
		WithExec([]string{"pip", "install", "-r", "requirements.txt"}) //ensures container has needed libraries to run python script
}

func main() {
	train()
}

func train() {
	ctx := context.Background()

	client, err := dagger.Connect(ctx)
	if err != nil {
		log.Fatalf("Failed to connect to Dagger: %v", err)
	}
	defer client.Close()

	// Load entire repo from project root
	project := client.Host().Directory("..") // mount entire repo (all files are accessible)

	//set up base container:
	pipeline := runPython(ctx, client, project)

	// Data prep and cleaining
	log.Println("Data prep and cleaning...")
	pipeline = pipeline.WithExec([]string{
		"python", "mlops_src/templ_dataset.py",
	})

	// Feature selection
	log.Println("Feature selection...")
	pipeline = pipeline.WithExec([]string{
		"python", "mlops_src/templ_features.py",
	})

	// Model training
	//log.Println("Training...")
	//pipeline = pipeline.WithExec([]string{
	//	"python", "mlops_src/modeling/templ_train.py",
	//})

	// Force evaluation to ensure the script runs and the model file is created
	_, err = pipeline.Sync(ctx)
	if err != nil {
		// Check if it's a dagger Exec error
		if execErr, ok := err.(*dagger.ExecError); ok {
			// Print both stdout and stderr
			log.Printf("Command failed:\nSTDOUT:\n%s\nSTDERR:\n%s", execErr.Stdout, execErr.Stderr)
		} else {
			log.Fatalf("ML pipeline execution failed: %v", err)
		}
	}

	log.Println("First 2 steps completed successfully.")

	// Saving model artifacts
	//log.Println("Exporting model artifact …")

	//modelFile := pipeline.File("models/lead_model.pkl")

	//_, err = modelFile.Export(ctx, "../models/model.pkl") //take only this file from the container (there are other aritfacts)
	//if err != nil {
	//	log.Fatalf("Failed to export trained model: %v", err)
	//}
	//log.Println("Export complete")

}

func test() {
	ctx := context.Background()

	client, err := dagger.Connect(ctx)
	if err != nil {
		log.Fatalf("Failed to connect to Dagger: %v", err)
	}
	defer client.Close()

	project := client.Host().Directory("..")

	pipeline := runPython(ctx, client, project)

	pipeline = pipeline.WithExec([]string{
		"python", "mlops_src/modeling/templ_predict.py",
	})
}
