# Anomaly Detection using Random Forest (From Scratch)

## Overview

This project implements a **Random Forest algorithm from scratch in Python** for anomaly detection / classification using a CSV dataset for Diabetic Retinopathy.

## Dataset : https://www.kaggle.com/c/diabetic-retinopathy-detection/data

Instead of using machine learning libraries such as **scikit-learn**, the algorithm is implemented manually to demonstrate how Random Forest works internally.

The implementation includes:

* Decision Tree construction
* Gini index calculation
* Bootstrap sampling
* Bagging (ensemble learning)
* Cross-validation evaluation

---

## Features

* Random Forest implementation from scratch
* Decision tree learning using Gini impurity
* Bootstrap sampling for training trees
* Majority voting for prediction
* Cross-validation model evaluation
* Probability estimation for predictions

---

## Project Structure

```
Anomaly_Detection_Random_Forest/
│
├── main.py            # Main script containing the Random Forest implementation
├── train.csv          # Input dataset
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

---

## Requirements

The project requires Python 3 and the following library:

* numpy

Install dependencies using:

```
pip install -r requirements.txt
```

---

## Dataset

The dataset is loaded from:

```
train.csv
```

The CSV file should contain:

* Feature columns
* A **target label in the last column**

Example format:

```
feature1,feature2,feature3,label
5.1,3.5,1.4,0
6.2,3.4,5.4,1
4.9,3.1,1.5,0
```

Labels used in the dataset:

* 0
* 1
* 2 (limited sampling in loader)

---

## Algorithm Workflow

1. Load dataset from CSV file
2. Perform **cross-validation split**
3. Train multiple **decision trees**
4. Each tree uses:

   * Random feature selection
   * Bootstrap sampling
5. Trees make predictions independently
6. Final prediction is made by **majority voting**

---

## Model Parameters

| Parameter   | Description                          |
| ----------- | ------------------------------------ |
| n_folds     | Number of folds for cross validation |
| max_depth   | Maximum depth of decision tree       |
| min_size    | Minimum samples required to split    |
| sample_size | Ratio of dataset used for each tree  |
| n_trees     | Number of trees in forest            |
| n_features  | Number of random features per split  |

Example values used:

```
n_folds = 5
max_depth = 10
min_size = 1
sample_size = 1.0
n_trees = 5
```

---

## Running the Project

Run the script using:

```
python main.py
```

Example output:

```
Trees: 5
Scores: [88.0, 90.0, 87.5, 91.2, 89.4]
Mean Accuracy: 89.22%
```
## Results Observed :
For our experiment no of trees are 5 and max depth of a tree is 10. 
In pixel based approach number of training instances were 400 and testing were 100. 
In feature based approach total number of instances were 1157 out of which 921 were training set and 230 were testing set.
Accuracy :
Pixel based approach 70.556%
Feature based approach in random forest 62.5%

---

## Key Concepts Implemented

### Gini Index

Used to evaluate the quality of dataset splits in decision trees.

### Bootstrap Aggregation (Bagging)

Each tree is trained on a random sample of the dataset.

### Random Feature Selection

At each split, a random subset of features is used to reduce correlation between trees.

---

## Educational Purpose

This project is useful for understanding:

* How **Random Forest works internally**
* Decision tree learning
* Ensemble learning techniques
* Model evaluation with cross-validation

---

## License

This project is released under the **MIT License**.





