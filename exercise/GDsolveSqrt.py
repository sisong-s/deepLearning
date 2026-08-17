def gd_sqrt(x):
    lr = 0.01
    epochs = 1000
    w = 1.0  # Initial guess for the square root
    for _ in range(epochs):
        pred = w * w  # Current prediction of the square
        error = pred - x  # Difference from the actual value
        dw = 2 * error * w  # Derivative of the loss with respect to w
        w -= lr * dw  # Update w using gradient descent
    return w


x = 4.0
ans = gd_sqrt(x)
print(ans)