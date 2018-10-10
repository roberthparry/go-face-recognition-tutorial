package main

import (
	"fmt"
	"log"
	"path/filepath"

	"github.com/Kagami/go-face"
)

const dataDir = "images"

func main() {
	fmt.Println("Facial Recognition System v0.01")

	rec, err := face.NewRecognizer(dataDir)
	if err != nil {
		fmt.Println("Error Thrown Creating New Recognizer")
		fmt.Println(err)
	}
	defer rec.Close()

	avengersImage := filepath.Join(dataDir, "avengers-02.jpeg")

	faces, err := rec.RecognizeFile(avengersImage)
	if err != nil {
		log.Fatalf("can't recognize file")
	}
	fmt.Println("Number of faces in image: ", len(faces))

	var samples []face.Descriptor
	var avengers []int32
	for i, f := range faces {
		samples = append(samples, f.Descriptor)
		avengers = append(avengers, int32(i))
	}

	labels := []string{
		"Dr Strange",
		"Tony Stark",
		"Bruce Banner",
		"Wong",
	}

	rec.SetSamples(samples, avengers)

	testFace := filepath.Join(dataDir, "wong.jpg")
	tonyStark, err := rec.RecognizeSingleFile(testFace)
	if err != nil {
		log.Fatalf("Faced error with file: %v", err)
	}

	avengerId := rec.Classify(tonyStark.Descriptor)
	if avengerId < 0 {
		log.Fatalf("Can't Classify based off existing database")
	}

	fmt.Println(avengerId)
	fmt.Println(labels[avengerId])
}
