class Model:
    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs
        self.w = 0.0
        self.b = 0.0

    def forward(self, x):
        return self.w*x+self.b

    def compute_gradient(self):
        dw = 0
        db = 0
        for x,y in zip(x_data, y_data):
            pred = self.forward(x)
            error = pred-y
            dw += error*x
            db += error
        n = len(x_data)
        return dw/n, db/n
    def fit(self, x_data, y_data):
        for _ in range(self.epochs):
            dw, db = self.compute_gradient()
            self.w -= self.lr*dw
            self.b -= self.lr*db
        return self.w, self.b

if __name__ == "__main__":
    x_data = [1,2,3,4]
    y_data = [3.1,5.1,6.9,8.8]

    model = Model(lr=0.01, epochs=1000)
    w, b = model.fit(x_data, y_data)
    print(w,b)