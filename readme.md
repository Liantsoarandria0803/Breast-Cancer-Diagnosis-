

# Decision Tree Classifier in Python

This project implements a **Decision Tree Classifier** from scratch in Python. The implementation is designed to handle classification tasks using simple, interpretable decision trees. The project does not rely on machine learning libraries like `scikit-learn` for the core logic, demonstrating how a decision tree can be built from the ground up.

## Features

- Handles classification tasks.
- Implements custom splitting logic based on **information gain**.
- Supports stopping criteria based on:
  - Maximum depth of the tree.
  - Minimum number of samples required to split a node.
- Includes a prediction method to classify new data points.
- Calculates **entropy** for assessing the quality of splits.

## Requirements

The following Python libraries are required to run the code:

- `numpy`
- `sklearn` (for dataset and train-test splitting)

Install them using:

```bash
pip install numpy scikit-learn
```


## Implementation Details

### Classes and Methods

1. **Node**: Represents a single node in the decision tree.
   - Stores information about:
     - Feature used for the split.
     - Threshold value.
     - References to child nodes.
     - Leaf node classification.

2. **DecisionTree**: Main class for building and using the decision tree.
   - `fit(X, y)`: Fits the model to the training data.
   - `predict(X)`: Predicts class labels for the given input data.
   - `_grow_tree(X, y, depth)`: Recursively builds the tree.
   - `_best_split(X, y, feat_idx)`: Finds the best feature and threshold for splitting.
   - `_information_gain(y, X_column, threshold)`: Calculates the information gain for a split.
   - `_entropy(y)`: Computes the entropy of a dataset.

### Key Features

- **Entropy and Information Gain**: Splits are chosen based on the reduction in entropy (information gain).
- **Stopping Criteria**: The recursion stops when:
  - The maximum depth is reached.
  - The number of samples at a node is less than `min_samples_split`.
  - All samples belong to the same class.

## Example Usage

The code includes an example using the Breast Cancer dataset from `scikit-learn`:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Create and train Decision Tree
clf = DecisionTree()
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate accuracy
accuracy = np.sum(y_test == y_pred) / len(y_test)
print("Accuracy:", accuracy)
```

### Output
The classifier predicts the labels for the test set and computes the accuracy. Example output:

```
Accuracy: 0.9473684210526315
```

## How to Run

1. Clone the repository or copy the code files to your local machine.
2. Install the required dependencies.
3. Run the example script to train and evaluate the decision tree.


## Notes

- This implementation is intended for educational purposes and may not be optimized for large datasets or real-world use cases.
- If you need high-performance models, consider using libraries like `scikit-learn`.

## Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request.

## License

This project is open-source and available under the MIT License. See the `LICENSE` file for details.

---

Feel free to modify this template to suit your specific needs.