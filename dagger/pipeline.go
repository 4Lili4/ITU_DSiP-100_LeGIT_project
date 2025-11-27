package main

import (
	"context"
	"log"

	"dagger.io/dagger"
)

func runPython(ctx context.Context, client *dagger.Client, src *dagger.Directory, script string) *dagger.Container {
	return client.Container().
		From("python:3.10").
		WithDirectory("/workspace", src).
		WithWorkdir("/workspace").
		WithExec([]string{"python", script})
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

	// Data prep and cleaining
	log.Println("Running I_data_prep.py …")
	step1 := runPython(ctx, client, project, "mlops_src/dataset-py/I_data_prep.py")

	log.Println("Running II_data_cleaning.py …")
	step2 := step1.WithExec([]string{"python", "mlops_src/dataset-py/II_data_cleaning.py"})

	// Data splitting
	log.Println("Running III_data_splitting.py …")
	step3 := step2.WithExec([]string{"python", "mlops_src/features-py/III_data_splitting.py"})

	// Model training and selection
	log.Println("Running IV_model_training.py …")
	trained := step3.WithExec([]string{
		"python", "mlops_src/modeling/train-py/IV_model_training.py"})

	log.Println("Running V_model_test.py …")
	tested := trained.WithExec([]string{
		"python", "mlops_src/modeling/train-py/V_model_test.py"})

	log.Println("Running VI_model_selection.py …")
	finalStep := tested.WithExec([]string{
		"python", "mlops_src/modeling/train-py/VI_model_selection.py"})

	// Saving model artifacts
	log.Println("Exporting model artifact …")

	modelFile := finalStep.File("final_model.pkl")

	_, err = modelFile.Export(ctx, "../models/model.pkl")
	if err != nil {
		log.Fatalf("Failed to export trained model: %v", err)
	}

}
