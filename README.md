# mlops_week1
# Iris Species Classifier with Decision Tree

This project demonstrates training a Decision Tree model on the Iris dataset and saving it for deployment, including usage in a custom container for serving predictions (e.g., on Google Cloud Vertex AI).

## Project Structure
```
.
├── data/
│   └── iris.csv                  # Input dataset
├── models/
│   └── decision\_tree\_model.pkl   # Saved model
├── train.py                      # Script to train and serialize model
├── SDK\_Custom\_Container\_Prediction.ipynb  # Notebook for prediction using custom container
└── README.md                     # Project documentation
```

## Dataset

- **Source**: Iris Dataset (commonly used for classification tasks)
- **Features**: 
  - `sepal_length`
  - `sepal_width`
  - `petal_length`
  - `petal_width`
- **Target**: 
  - `species` (Setosa, Versicolor, Virginica)

## Model

- **Algorithm**: `DecisionTreeClassifier` from scikit-learn
- **Max Depth**: 3
- **Train-Test Split**: 60% train / 40% test (stratified by species)
- **Accuracy**: Printed after evaluation on test data

## Setup

### Prerequisites

- Python 3.7+
- pip

### Install Dependencies

```bash
pip install pandas numpy scikit-learn joblib
````

### Prepare the Data

Ensure `iris.csv` is located inside the `data/` directory:

```bash
mkdir -p data
# place iris.csv in data/
```

### Run Training Script

```bash
python train.py
```

* Trains a decision tree on the Iris dataset.
* Prints test accuracy.
* Saves the model to `models/decision_tree_model.pkl`.

## Custom Container for Prediction (Vertex AI)

The notebook `SDK_Custom_Container_Prediction.ipynb` demonstrates how to:

* Create a Docker container that serves predictions using the trained model.
* Push the image to Google Container Registry (GCR).
* Deploy the model using Google Cloud Vertex AI SDK.
* Send prediction requests to the deployed model.

## Model Usage

To use the model for predictions locally:

```python
import pickle
import numpy as np

model = pickle.load(open("models/decision_tree_model.pkl", "rb"))
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = model.predict(sample)
print(prediction)
```
