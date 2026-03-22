class Model:
    def __init__(self,lr,epochs):
        self.lr = lr
        self.epochs = epochs
        self.w = 0.0
        self.b = 0.0
    
    def forward(self, x):
        return self.w * x + self.b
    # 这里是batchsize = 样本数
    def fit(self, X, y):
        for _ in range(self.epochs):
            dw = 0.0
            db = 0.0
            for xi,yi in zip(X,y):
                pred = self.forward(xi)
                error = pred - yi
                dw += error * xi
                db += error
            self.w -= self.lr*dw
            self.b -= self.lr*db

    def predict(self, X):
        preds = []
        for xi in X:
            preds.append(self.forward(xi))
        return preds

X = [1,2,3,4]
y = [3.1,5.1,6.9,8.8]
model = Model(0.001, 1000)
model.fit(X,y)
print(model.predict(X))