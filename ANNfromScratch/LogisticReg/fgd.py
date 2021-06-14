from utils import process_data, yreshape, initparams, forwardprop, gradW, gradb, cost, error_rate
import numpy as np

Xtrain, Xtest, Ytrain, Ytest = process_data()
Ytrain_ind = yreshape(Ytrain)
Ytest_ind = yreshape(Ytest)
N,D = Xtrain.shape
K = Ytrain_ind.shape[1]
print("Number of unique class: ",K)

W, b, lr, reg, n_iters = initparams(D,K)
W0 = W.copy()
lr = 0.9

for i in range(50):

    p_y = forwardprop(Xtrain,W,b)
    train_loss = cost(Ytrain_ind,p_y)
    train_err = error_rate(Ytrain_ind,p_y)

    gW = gradW(Ytrain_ind,p_y,Xtrain) / N
    gb = gradb(Ytrain_ind,p_y) / N
    W -= lr*(gW - reg*W)
    b -= lr*(gb - reg*b)

    p_ytest = forwardprop(Xtest,W,b)
    test_loss = cost(Ytest_ind,p_ytest)
    test_err = error_rate(Ytest_ind,p_ytest)

    if (i + 1) % 10 == 0:
        print(f"Iter: {i + 1}/{n_iters}, Train loss: {train_loss:.3f} "
              f"Train error: {train_err:.3f}, Test loss: {test_loss:3f} "
              f"Test error: {test_err:.3f}")

p_y = forwardprop(Xtest, W, b)
print("Final error rate for full GD: ", error_rate(Ytest_ind, p_y))