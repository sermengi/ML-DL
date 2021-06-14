from utils import process_data, yreshape, initparams, forwardprop, gradW, gradb, cost, error_rate
import numpy as np

def shuffle(X,Y):
    data = np.concatenate((X,Y),axis=1)
    np.random.shuffle(data)
    X = data[:,:784]
    Y = data[:,784:]
    return X,Y


Xtrain, Xtest, Ytrain, Ytest = process_data()
Ytrain_ind = yreshape(Ytrain)
Ytest_ind = yreshape(Ytest)
N,D = Xtrain.shape
K = Ytrain_ind.shape[1]
print("Number of unique class: ",K)

W, b, lr, reg, n_iters = initparams(D,K)
lr = 0.001

for i in range(50):
    Xtemp, Ytemp = shuffle(Xtrain,Ytrain_ind)
    for n in range(N):
        x = Xtemp[n,:].reshape(1,D)
        y = Ytemp[n,:].reshape(1,K)
        p_y = forwardprop(x,W,b)
        gW = gradW(y,p_y,x)
        gb = gradb(y,p_y)
        W -= lr*(gW - reg*W)
        b -= lr*(gb - reg*b)

    p_y = forwardprop(Xtrain,W,b)
    train_loss = cost(Ytrain_ind,p_y)
    train_err = error_rate(Ytrain_ind,p_y)
    p_ytest = forwardprop(Xtest,W,b)
    test_loss = cost(Ytest_ind,p_ytest)
    test_err = error_rate(Ytest_ind,p_ytest)

    print(f"Iter: {i + 1}/{n_iters}, Train loss: {train_loss:.3f} "
                  f"Train error: {train_err:.3f}, Test loss: {test_loss:3f} "
                  f"Test error: {test_err:.3f}")













