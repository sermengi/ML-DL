from util import spiral_data, shuffle, ANN_model
import numpy as np

# import the dataset
X, Y = spiral_data()

# shuffle the data
X, Y = shuffle(X, Y)

# Train Test Split
# 70% of samples are used to train the model
Ntrain = int(0.7*X.shape[0])
Xtrain, Xtest, Ytrain, Ytest = X[:Ntrain], X[Ntrain:], Y[:Ntrain], Y[Ntrain:]
_, D = Xtrain.shape
K = len(set(Y.flat))
print("Number of trained example: ", Ntrain)
print("Number of features: ",D)
print("Number of unique class: ",K)

# defining the hyper-parameters
M = 20
hidden_size = 2
log_lr = -4
log_r2 = -2
max_tries = 30

best_validation_rate = 0
best_hls = None
best_lr = None
best_l2 = None
best_M = None

for _ in range(max_tries):
    hls = [M] * hidden_size
    lr = 10 ** log_lr
    reg = 10 ** log_r2
    trainacc, valacc = ANN_model(D,hls,lr,reg,Xtrain,Ytrain,Xtest,Ytest)
    print(f"validation accuracy: {valacc:.3f}, train accuracy: {trainacc:.3f}, "
          f"settings: layer_size: {hidden_size}, layer_wide: {M}, learning rate: {lr}, regularization: {reg}")
    if valacc > best_validation_rate:
        best_validation_rate = valacc
        best_M = M
        best_hls = hidden_size
        best_lr = log_lr
        best_l2 = log_r2
    # select new parameter randomly
    hidden_size = best_hls + np.random.randint(-1,2)
    hidden_size = max(1,hidden_size)
    M = best_M + np.random.randint(-1, 2)*10
    M = max(10, M)
    log_lr = best_lr + np.random.randint(-1, 2)
    log_r2 = best_l2 + np.random.randint(-1, 2)

print(f"Best validation accuracy: {best_validation_rate}")
print(f"Optimum Hyperparameters: ")
print(f"hidden layer size: {best_hls}")
print(f"layer width: {best_M}")
print(f"learning rate: {best_lr}")
print(f"l2 value: {best_l2}")








