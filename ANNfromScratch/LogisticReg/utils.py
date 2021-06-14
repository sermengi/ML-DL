import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random

def process_data():

    path = "C:/Users/user/OneDrive/Masaüstü/Programming/Python/ANNfromScratch/LogisticReg"
    if not os.path.exists(path+"/large_files/train.csv"):
        print("Looking for ../large_files/train.csv")
        print("Cannot find the data")
        print("Check the train.csv location")

    # Load Data from train.csv file
    df = pd.read_csv(path+"/large_files/train.csv")
    data = df.to_numpy().astype(np.float32)

    # Shuffle the data
    np.random.shuffle(data)
    X = data[:,1:]
    Y = data[:,0]

    # Train and Test Split
    Xtrain = X[:-1000]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:]
    Ytest = Y[-1000:]
    print("Number of train samples: ",Xtrain.shape[0])
    print("Number of test samples: ",Xtest.shape[0])

    # Normalize the data
    mu = Xtrain.mean(axis=0)
    std = Xtrain.std(axis=0)

    # Check for zero std for ZeroDivisionError
    idx = np.where(std == 0)[0]
    assert np.all(std[idx] == 0)
    np.place(std,std == 0,1)

    # Normalize the train and test data
    Xtrain = (Xtrain-mu)/std
    Xtest = (Xtest-mu)/std

    return Xtrain,Xtest,Ytrain,Ytest

def yreshape(yarr):

    # Reshape the Ytrain and Ytest: (N,) --> (N,K)
    rows, cols = yarr.shape[0],int(yarr.max()+1)
    arr = np.zeros((rows, cols))
    for idx in range(rows):
        arr[idx][int(yarr[idx])] = 1

    return arr

def initparams(D,K):

    # Initialize the weights and biases for one neuron
    W = np.random.randn(D, K) / np.sqrt(D)
    b = np.zeros(K)

    lr = 0.00003
    reg = 0.0
    n_iters = 100

    return W,b,lr,reg,n_iters

def forwardprop(X,W,b):
    # Forward propagation f((X*W)+b)
    return softmax(np.dot(X,W) + b)

def softmax(num):
    # Softmax activation
    expnum = np.exp(num)
    return expnum / expnum.sum(axis=1,keepdims=True)

def cost(y,ypred):
    # Categorical Cross Entropy
    return -(y*np.log(ypred)).sum() / (y.shape[0])

def error_rate(y,ypred):
    predict = ypred.argmax(axis=1)
    yact = y.argmax(axis=1)
    return np.mean(predict != yact)

def gradW(y,ypred,x):
    return x.T.dot(ypred-y)

def gradb(y,ypred):
    return (ypred-y).sum(axis=0)


def main():

    Xtrain, Xtest, Ytrain, Ytest = process_data()
    Ytrain_ind = yreshape(Ytrain)
    Ytest_ind = yreshape(Ytest)
    N,D = Xtrain.shape
    K = Ytrain_ind.shape[1]
    print("Number of classes: ",K)

    # Initializing the parameters
    W,b,lr,reg,n_iters = initparams(D,K)
    batch_size = 150
    train_losses = []
    test_losses = []
    train_clasification_errors = []
    test_clasification_errors = []

    for i in range(n_iters):

        # Train and Test cost/error Calculation
        p_y = forwardprop(Xtrain,W,b)
        train_loss = cost(Ytrain_ind,p_y)
        train_losses.append(train_loss)

        train_err = error_rate(Ytrain_ind,p_y)
        train_clasification_errors.append(train_err)

        p_ytest = forwardprop(Xtest,W,b)
        test_loss = cost(Ytest_ind,p_ytest)
        test_losses.append(test_loss)

        test_err = error_rate(Ytest_ind,p_ytest)
        test_clasification_errors.append(test_err)

        # Updating W and b using Full Gradient Descent
        W -= lr*(gradW(Ytrain_ind,p_y,Xtrain) + reg*W)
        b -= lr*(gradb(Ytrain_ind,p_y))

        if (i + 1) % 10 == 0:
            print(f"Iter: {i+1}/{n_iters}, Train loss: {train_loss:.3f} "
                  f"Train error: {train_err:.3f}, Test loss: {test_loss:3f} "
                  f"Test error: {test_err:.3f}")

    # Final Prediction of Model
    p_y = forwardprop(Xtest,W,b)
    print("Final error rate: ",error_rate(Ytest_ind,p_y))

    plt.plot(train_losses,label="Train loss")
    plt.plot(test_losses,label="Test loss")
    plt.title("Loss per iteration")
    plt.legend()
    plt.show()

    plt.plot(train_clasification_errors, label="Train error")
    plt.plot(test_clasification_errors, label="Test error")
    plt.title("Classification Error per iteration")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()




