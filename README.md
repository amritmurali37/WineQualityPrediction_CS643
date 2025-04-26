# AWS Project Assignment 2 - CS643

This project trains and evaluates a logistic regression model to predict wine quality using Apache Spark on a 5-node EMR cluster on AWS.

## Contents
- `TrainModel.java`: Code to train a logistic regression model.
- `PredictModel.java`: Code to load the model and predict validation data, and calculate F1 Score.
- `pom.xml`: Maven build configuration.
- `Dockerfile`: Container setup to build the project (optional for local builds).

## Setup and Execution
1. Launch a 5-node EMR cluster with Hadoop and Spark installed.
2. Transfer the training and validation datasets to HDFS.
3. Submit Spark jobs using:
   - `TrainModel` for training.
   - `PredictModel` for prediction and F1 score calculation.

## Final Results
- **F1 Score Achieved**: 0.5672

## Author
amritmurali37
