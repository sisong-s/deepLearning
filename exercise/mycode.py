import math
class LR:
    def __init__(self, lr, epochs,dim):
        self.lr = lr
        self.dim = dim
        self.w = [0.0]*dim
        self.epochs = epochs
        self.b = 0.0

    def sigmoid(self,x):
        return 1.0 / (1.0 + math.exp(x))

    def forward(self,x):
        z = self.b
        for i in range(len(x)):
            z += self.w[i] * x[i]
        return self.sigmoid(z)

    def fit(self,x,y):
        for i in range(self.epochs):
            for xi,yi in zip(x,y):
                pred = self.forward(xi)
                target = 0 if yi == -1 else 1
                error = pred - target
                self.b -= self.lr*error
                for i in range(self.dim):
                    self.w[i] -= self.lr*error*xi[i]

    def predict(self,x):
        preds = []
        for xi in x:
            if(self.forward(xi)) >=0.5:
                preds.append(1)
            else:
                preds.append(-1)
        return preds

if __name__ == '__main__' :
    x = [[1,-1],[1,1],[-1,-1],[-1,1]]
    y = [1,1,-1,-1]
    lr = LR(lr=0.01, epochs=1000, dim=2)
    lr.fit(x,y)
    print(lr.predict(x))