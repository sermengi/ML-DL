from util import spiral_data, shuffle, ANN_model

# Generating the data
X, Y = spiral_data()

# Shuffle the data
X, Y = shuffle(X,Y)

# Train Test Split
# 70% of samples are used to train the model
Ntrain = int(0.7*X.shape[0])
Xtrain, Xtest, Ytrain, Ytest = X[:Ntrain], X[Ntrain:], Y[:Ntrain], Y[Ntrain:]
_, D = Xtrain.shape
K = len(set(Y.flat))
print("Number of trained example: ", Ntrain)
print("Number of features: ",D)
print("Number of unique class: ",K)

# Hyper-parameters to test
Hidden_layer_sizes = [[300],
                    [100,100],
                    [50,50,50]]
learning_rates = [1e-4,1e-3,1e-2]
l2_penalties = [0.,0.1,1.0]

# Grid search loop through all parameters
best_validation_rate = 0
best_hls = None
best_lr = None
best_l2 = None
for hls in Hidden_layer_sizes:
    for lr in learning_rates:
        for reg in l2_penalties:
            trainacc, valacc = ANN_model(D, hls, lr, reg, Xtrain, Ytrain, Xtest, Ytest)
            print(f"validation accuracy: {valacc:.3f}, train accuracy: {trainacc:.3f}, "
                  f"settings: layer_size: {hls[0]}, learning rate: {lr}, regularization: {reg}")
            if valacc > best_validation_rate:
                best_validation_rate = valacc
                best_hls = hls[0]
                best_lr = lr
                best_l2 = reg

print(f"Best validation accuracy: {best_validation_rate}")
print(f"Optimum Hyperparameters: ")
print(f"hidden layer size: {best_hls}")
print(f"learning rate: {best_lr}")
print(f"l2 value: {best_l2}")














