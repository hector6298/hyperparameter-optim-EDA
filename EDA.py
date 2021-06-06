import numpy as np
import sklearn
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

#==================================================================
#TODO change dtypes for every specific hyperparameter
#TODO Make changes to accept categorical hyperparameters
#TODO add verbosity parameter and implement logs accordingly
#TODO implement data split when folds=1
#TODO Implement hyperparameter boundaries support for common ML models (currently only supports RF)
#TODO implement multiprocessing/multithreading
#==================================================================



# Class to store a concrete set of parameters and store the metric (error)
class ParamObj(object):
    def __init__(self, paramDict:dict):
        self.paramDict = paramDict
        self.metric = None


def edaMLOpt(X:np.array, 
             Y:np.array, 
             grid:dict, 
             mins_dict:dict,
             dtypes:list,
             nR:int,
             nN:int,
             nIterNum:int,
             edaThreshold:float,
             folds:int=3):
    """
    Perform EDA evolutionary algorithm for optimizing ML models. TODO (Not quite ready yet!!!)

    Parameters:

        X: Numpy array. Feature matrix of shape (samples, features)
        Y: Numpy arrray. Categorical Labels
        grid: Dict. Starting grid of hyperparameters
        mins_dict: Dict. Minimum values for the search of hyperparameters. Must have the same keys as grid, in the sasme order.
        dtypes: List of dtypes. The datatype of each hyperparameter in the same order as the keys of grid.
        fold: Int. How many folds in cross validation
        nR: Int. Initial population
        nN: Int. How many individuals will survive per generation. Others will be eliminated.
        nIterNum: Int. Maximum number of generations
    
    Returns:

        Parameters: dict. Parameters with minimal loss of accuracy on the optimization set.
    """

    # Make sure we have the same keys
    assert [*mins_dict] == [*grid], "Both the grid and the minimus must have the same keys"

    # utility function to get random initial parameters
    def initParam(paramGrid:dict, random_state=42):
        paramDict = dict()
        for key in paramGrid:
            paramDict[key] = np.random.choice(paramGrid[key])
        return paramDict    

    # utility function to estimate new param grid
    def estNewParamGrid(paramSet:list, mins_dict:dict, dtypes:list):
        iparam = 0; nParamNum = len(paramSet[0].paramDict.keys()); iparam=0; numInSet = len(paramSet)

        newGrid = dict()

        # Get the mean of all parameters
        paramMeans = [0]*nParamNum
        for i in range(len(paramSet)):
            keys = list(paramSet[i].paramDict.keys())
            for j in range(len(keys)):
                paramMeans[j] += paramSet[i].paramDict[keys[j]]
        
        for j in range(nParamNum):
            paramMeans[j] /= numInSet
        
        paramVariances = [0]*nParamNum
        for i in range(len(paramSet)):
            keys = list(paramSet[i].paramDict.keys())
            for j in range(len(keys)):
                paramVariances[j] += pow(paramSet[i].paramDict[keys[j]] - paramMeans[j], 2)
        
        for j in range(nParamNum):
            paramVariances[j] /= numInSet
            newGrid[keys[j]] = np.linspace(max(mins_dict[keys[j]],paramMeans[j] - paramVariances[j]), (paramMeans[j] + paramVariances[j]), dtype=dtypes[j])

        return newGrid

    # Max starting float possible
    errMeanPrev = sys.float_info.max

    iIter = 0

    # Initializing parameter population

    if nN > nR:
        raise Exception("Error: Selected population should be less than initial population")

    paramSet = []

    for iR in range(nR):
        paramDict = initParam(grid)
        paramObj = ParamObj(paramDict)
        paramSet.append(paramObj)

    print("Start EDA optimization for camera calibration")

    # Hold metrics for further visualization
    generationalErrors = []
    generationalStds = []

    #Start of the actual optimization
    while nIterNum > iIter:
        print(f"==== generation {iIter}: ====")
        iProc = 0
        bProc25 = False
        bProc50 = False
        bProc75 = False
        errMean = 0.0
        errStd = 0.0

        #Calculate all metrics for each individual paramSet
        for i in range(len(paramSet)):

            accuracyKFolds = 0.0

            kf = KFold(n_splits=folds)
            j = 0
            for train_index, test_index in kf.split(X):
                print(f"fitting fold {j} of individual {i}")
                #get samples for train and test
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = Y[train_index], Y[test_index]

                # fit the model for each fold
                rfp = paramSet[i].paramDict
                clf = RandomForestClassifier(**rfp).fit(X_train, y_train)
                preds = clf.predict(X_test)
                accuracy = accuracy_score(y_test, preds, normalize=True)

                # Aggregate metric
                accuracyKFolds += accuracy

                j+=1

            # get the average and assign to object
            accuracyKFolds /= folds
            print(f"Accuracy for ind. {i} -> {accuracyKFolds}")
            error = 1 - accuracyKFolds
            paramSet[i].metric = error

            # Add for global metric average
            errMean += error
            iProc += 1

            # Of course we check progress...
            if ((float(iProc) / float(nR)) > 0.25) and (not bProc25):
                print("25%%...")
                bProc25 = True
            if ((float(iProc) / float(nR)) > 0.50) and (not bProc50):
                print("50%%...")
                bProc50 = True
            if ((float(iProc) / float(nR)) > 0.75) and (not bProc75):
                print("75%%...")
                bProc75 = True
        
        # Finish computing global metric average
        errMean /= nR
        generationalErrors.append(errMean)

        # Get metrics global standard deviation
        for iparam in paramSet:
            error = iparam.metric
            errStd += pow(error - errMean,2)
        errStd = np.sqrt(errStd / nR)
        generationalStds.append(errStd)

        print("100%%!")
        print(f"current error mean = {errMean}")
        print(f"current error standard deviation = {errStd}")

        #check if generation needs to stop
        if (nIterNum < iIter) and ((errMeanPrev * edaThreshold) > np.abs(errMean - errMeanPrev)):
            print("Error is small enough. Stop generation.")
            break
    
        errMeanPrev = errMean

        # Sort parameters by their loss in accuracy (1-accuracy) and only keep nN individuals
        paramSet = sorted(paramSet, key=lambda param: param.metric)
        paramSet = paramSet[:nN]

        # Give rise to a new generation!!! 
        grid = estNewParamGrid(paramSet, mins_dict, dtypes)
        for iR in range(nR):
            params = initParam(grid)
            paramObj = ParamObj(params)
            paramSet.append(paramObj)

        iIter += 1
        print("\n")
            
    if nIterNum <= iIter:
        print("Exit: results cannot converge")

    return paramSet[0], generationalErrors, generationalStds