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
lr = 0.08
batch_size = 500
num_of_batch = int(np.ceil(N/batch_size))

for n in range(10):
    Xtemp, Ytemp = shuffle(Xtrain,Ytrain_ind)
    for i in range(num_of_batch):
        x = Xtemp[i*batch_size:(i+1)*batch_size,:]
        y = Ytemp[i*batch_size:(i+1)*batch_size,:]
        p_y = forwardprop(x,W,b)
        current_batch_size = len(x)
        gW = gradW(y,p_y,x) / current_batch_size
        gb = gradb(y,p_y) / current_batch_size
        W -= lr*(gW - reg*W)
        b -= lr*(gb - reg*b)

        p_y = forwardprop(Xtrain,W,b)
        train_loss = cost(Ytrain_ind,p_y)
        train_err = error_rate(Ytrain_ind,p_y)
        p_ytest = forwardprop(Xtest,W,b)
        test_loss = cost(Ytest_ind,p_ytest)
        test_err = error_rate(Ytest_ind,p_ytest)

    print(f"Iter: {n + 1}/{n_iters}, Train loss: {train_loss:.3f} "
                  f"Train error: {train_err:.3f}, Test loss: {test_loss:3f} "
                  f"Test error: {test_err:.3f}")