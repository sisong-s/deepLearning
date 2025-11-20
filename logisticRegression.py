import math
class LogisticRegression:
    def __init__(self, lr, epochs, dim):
        self.lr = lr
        self.epochs = epochs
        self.dim = dim
        self.b = 0.0
        self.w = [0.0] * dim

    def sigmoid(self, z):
        return 1.0 / (1.0 + math.exp(-z))

    def forward(self, x):
        z = self.b
        for i in range(len(x)):
            z += self.w[i] * x[i]
        return self.sigmoid(z)

    def fit(self, X, y):
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                p = self.forward(xi)
                target = 1 if yi ==-1 else 0
                error = p - target
                self.b -= self.lr * error
                for i in range(self.dim):
                    self.w[i] -= self.lr * error * xi[i]
    
    def predict(self, x):
        preds = []
        for xi in x:
            if self.forward(xi) >= 0.5:
                preds.append(1)
            else:
                preds.append(-1)
        return preds
    
if __name__ == '__main__':
    X = [[1,1],[1,-1],[-1,1],[-1,-1]]
    y = [-1,-1,1,1]
    model = LogisticRegression(lr=0.01, epochs=1000, dim=2)
    model.fit(X,y)
    print(model.predict(X))