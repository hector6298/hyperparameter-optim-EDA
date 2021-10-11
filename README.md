# Estimation of Distribution Evolutionary Algorithm for Hyperparameter Optimization

Wow what a mouthful. Just a simple hyperparameter optimization algorithm based on EDA for machine learning classifiers.

## Getting Started

Clone the repository and install dependencies:

```
git clone https://github.com/hector6298/eda_opt.git
cd eda_opt
sudo pip3 install -r requirements.txt
```

Now you can use jupyter or Google Colab to run test.ipynb, or you can use the optimizer object on your own scripts like so:

1) Import the module

```
from EDA import edaMLOpt
```

2) Load your data. For now it should be a 2D Numpy array for the features and a vector for the targets. Example:

```
from sklearn.datasets import load_iris

iris_df = load_iris(as_frame=True)
dataframe = iris_df["frame"].sample(frac=1)

X = dataframe.iloc[:,:-1].values
Y = dataframe.iloc[:,-1].values
```

3) Define hyperparameter grid for your ML classifier. It should be a dict, in which each key contains the exact name of the hyperparameter and the values are, well, the hyperparameter's search space. Example:
```
#Set starting grid Point
grid = {
    "n_estimators" : np.linspace(10, 100, dtype=int),
    #"criterion" : ["gini", "entropy"],
    "min_samples_split" : np.linspace(2, 100, dtype=int),
    "min_samples_leaf": np.linspace(2, 100, dtype=int),
    "max_features": np.linspace(0.3, 1.0, num=7, dtype=float),   
}

# Minimums for random forest hyperparameters. Same key names as grid
mins_rf = {
    "n_estimators" : 1,
    "min_samples_split" : 2,
    "min_samples_leaf" : 2,
    "max_features" : 0.1
}

dtypes = [int, int, int, float]
```

4) Run the optimizer:
```
params = edaMLOpt(X,Y, grid=grid, mins_dict=mins_rf, dtypes=dtypes,
                  folds=3, nR=20, nN=4, nIterNum=4, 
                  edaThreshold=0.01)
```

That's it!

## TODOs:

#TODO Make changes to accept categorical hyperparameters
#TODO add verbosity parameter and implement logs accordingly
#TODO implement data split when folds=1
#TODO Implement hyperparameter boundaries support for common ML models (currently only supports RF)
