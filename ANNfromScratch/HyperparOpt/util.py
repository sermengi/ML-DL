import numpy as np
from keras.layers import Input, Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras import regularizers


def spiral_data():
    radius = np.linspace(1, 10, 100)
    thetas = np.empty((6, 100))
    for i in range(6):
        start_angle = np.pi * i / 3.0
        end_angle = start_angle + np.pi / 2.0
        points = np.linspace(start_angle, end_angle, 100)
        thetas[i] = points

    # Polar coordinate to cartesian coordinate
    x1 = np.empty((6, 100))
    x2 = np.empty((6, 100))
    for i in range(6):
        x1[i] = radius * np.cos(thetas[i])
        x2[i] = radius * np.sin(thetas[i])

    # Creating the input array
    X = np.empty((600, 2))
    X[:, 0] = x1.flatten()
    X[:, 1] = x2.flatten()

    # Adding noise to inputs
    X += np.random.rand(600, 2) * 0.5

    # Target values for inputs
    Y = np.array([0] * 100 + [1] * 100 + [0] * 100 + [1] * 100 + [0] * 100 + [1] * 100)

    return X, Y


def shuffle(X, Y):
    Y = np.expand_dims(Y, axis=1)
    data = np.concatenate((X, Y), axis=1)
    np.random.shuffle(data)
    X, Y = data[:, :2], data[:, 2:]
    return X, Y


def ANN_model(D,hls,lr,reg,Xtr,Ytr,Xte,Yte):

    num_hl = len(hls)
    num_neuron = hls[0]

    model = Sequential()
    model.add(Input(shape=(D,)))
    model.add(Dense(num_neuron, activation="relu", kernel_regularizer=regularizers.l2(reg)))
    if (num_hl == 2):
        model.add(Dense(num_neuron, activation="relu", kernel_regularizer=regularizers.l2(reg)))
        if (num_hl == 3):
            model.add(Dense(num_neuron, activation="relu", kernel_regularizer=regularizers.l2(reg)))

    model.add(Dense(1, activation="softmax"))
    opt = SGD(learning_rate=lr, momentum=0.99)

    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(Xtr, Ytr, epochs=3000, verbose=0)
    _, train_acc = model.evaluate(Xtr, Ytr,verbose=0)
    _, test_acc = model.evaluate(Xte, Yte,verbose=0)
    return train_acc, test_acc
