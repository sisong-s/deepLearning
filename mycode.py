import numpy as np

def gradient_descent(x,y,lr, epochs):
    w = 0.0
    b = 0.0
    n = len(x)

    for i in range(epochs):
        y_pred = w * x + b
        loss = np.mean((y - y_pred)**2)

        