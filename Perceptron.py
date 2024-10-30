import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
# This imports the necessary libraries:
# - `numpy` (as np): For efficient numerical calculations, especially with arrays and matrices.
# - `os`: A library to interact with the operating system (though it's not used in the current code).
# - `pandas` (as pd): For data manipulation and analysis, especially to handle datasets like CSV files.

class Perceptron:
    """Perceptron classifier.

    The Perceptron is a simple linear classifier, meaning it attempts to separate data points into two categories
    by drawing a straight line (or hyperplane in higher dimensions). It is one of the simplest types of neural networks.
    
    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0). Controls how much the model adjusts the weights with each update.
    n_iter : int
      Number of times the algorithm goes through the entire training dataset (also known as epochs).
    random_state : int
      Random number seed to ensure reproducibility (so the same random values are generated each time).

    Attributes
    -----------
    w_ : 1d-array
      Weights that the model will learn during training (initially set randomly).
    b_ : Scalar
      Bias term, a constant added to the net input to control the decision boundary.
    errors_ : list
      Keeps track of how many misclassifications happen at each epoch, used to track the learning process.
    """

    # The constructor (__init__) initializes the perceptron object with the learning rate, number of iterations, 
    # and a random state for reproducible randomness.
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta  # Learning rate
        self.n_iter = n_iter  # Number of iterations (epochs)
        self.random_state = random_state  # Random state to initialize weights reproducibly

    # The fit method is used to train the perceptron on the given data X (features) and y (labels).
    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
            Training data, where n_examples is the number of examples (rows), and n_features is the number of features (columns).
        y : array-like, shape = [n_examples]
            Target values (the correct labels for the data points).

        Returns
        -------
        self : object
        """
        # Initialize the random generator to set the initial weights to small random values close to 0.
        rgen = np.random.RandomState(self.random_state)
        
        # Randomly initialize the weights using a normal distribution (mean=0.0, std=0.01) for all features.
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        
        # Initialize the bias term (b_) to 0.
        self.b_ = np.float16(0.)
        
        # List to track the number of misclassifications during each epoch.
        self.errors_ = []

        # Loop over the dataset n_iter times (epochs).
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):  # Iterate over each training example and its target label.
                # Calculate the update for the weights based on the difference between the predicted and actual labels.
                update = self.eta * (target - self.predict(xi))
                
                # Update the weights and bias term. The weights are updated based on the input features (xi),
                # and the bias is updated by the calculated update.
                self.w_ += update * xi
                self.b_ += update
                
                # Track the number of errors (misclassifications) during this epoch.
                errors += int(update != 0.0)
            
            # Append the number of errors in this epoch to the errors list.
            self.errors_.append(errors)
        return self

    # The net_input method calculates the net input (weighted sum of inputs plus bias), which is used to make predictions.
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    # The predict method returns the predicted class label (1 or 0) based on the net input.
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)  # If net_input >= 0, predict class 1, otherwise predict 0.
    
# Example: A small vector calculation showing how to use numpy to compute the angle between two vectors.
v1 = np.array([1, 2, 3])  # Define a 3-element vector.
v2 = 0.5 * v1  # Scale v1 by 0.5 to create v2.

# Calculate the angle between v1 and v2 using the dot product formula.
np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

# Load the Iris dataset from a URL using pandas. The dataset is stored as a CSV file at this URL.
s = 'https://archive.ics.uci.edu/ml/' \
    'machine-learning-databases/iris/iris.data'

print('From URL:', s)

# Read the CSV data from the URL into a pandas DataFrame.
df = pd.read_csv(s,
                 header=None,  # No column headers in the dataset, so specify None.
                 encoding='utf-8')  # UTF-8 encoding to ensure proper reading of the data.

# Print the last 5 rows of the dataset to verify it was loaded correctly.
print(df.tail())


# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values
# plot data

def plot_decision_regions(X, y, classifier, resolution=0.02):
  # setup marker generator and color map
  markers = ('o', 's', '^', 'v', '<')
  colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
  cmap = ListedColormap(colors[:len(np.unique(y))])
  
  # plot the decision surface
  x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
  np.arange(x2_min, x2_max, resolution))
  lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
  lab = lab.reshape(xx1.shape)
  plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
  plt.xlim(xx1.min(), xx1.max())
  plt.ylim(xx2.min(), xx2.max())
  # plot class examples
  for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=X[y == cl, 0],
    y=X[y == cl, 1],
    alpha=0.8,
    c=colors[idx],
    marker=markers[idx],
    label=f'Class {cl}',
    edgecolor='black')

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('Sepal length CM')
plt.ylabel('Petal length CM')
plt.legend(loc='upper left')
plt.show()