#!/bin/bash

# Define the base path for data
BASE_PATH="../../data/shared_data"
RESULT_FILE="analysis_results.csv"

# Number of times to run the analysis
X=100  # Change this value to the number of times you want to run the code

# Function to run the Docker setup and analysis
run_analysis() {
    # Clear previous data
    rm -rf "$BASE_PATH"/*

    # Restart Docker containers
    docker-compose -f ../../docker-compose.server.yml down -v
    docker-compose -f ../../docker-compose.server.yml up --build -d

    docker-compose -f ../../docker-compose.client-training.yml down -v
    docker-compose -f ../../docker-compose.client-training.yml up --build &

    docker-compose -f ../../docker-compose.centralized-client-training.yml down -v
    docker-compose -f ../../docker-compose.centralized-client-training.yml up --build &

    wait

    docker-compose -f ../../docker-compose.client-prediction.yml up --build
    docker-compose -f ../../docker-compose.centralized-client-prediction.yml up --build

    # Run the Python analysis script
    python analysis.py
}

# Main loop to run the analysis X times
for ((i = 1; i <= X; i++)); do
    echo "Running analysis iteration $i..."
    run_analysis
done

echo "Analysis completed $X times."
