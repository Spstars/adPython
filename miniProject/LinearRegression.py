import numpy as np
import MakeDataset
from tqdm import tqdm
class LinearRegression:
    def __init__(self,iteration = 500,learning_rate=0.001) :
        self.lr = learning_rate
        self.iter = iteration
        self.weight=None
        self.bias =0
    def fit(self,X,y):
        #i only think about the case when "y = r1 * x + b"
        #if we have more rx, weight should be changed.
        
        n_feature,length = X.shape
        self.weight = np.ones(n_feature)
        print(X,X.shape)
        for _ in tqdm(range(self.iter)):
            y_pred =np.dot(self.weight,X)+ self.bias
            print("y_pred : ", y_pred)
            break
            dw = np.sum((1 /length ) * X * (y_pred-y))
            db = (1 /length) * np.sum(y_pred-y)
            self.weight -= dw * self.lr
            self.bias -=db * self.lr
        print("weight : ",self.weight)
    def predict(self,X):
        return np.dot(self.weight,X)+ self.bias ,self.weight,self.bias





if __name__ == "__main__":
    print("main")
    X,y, r1 ,b = MakeDataset.make_sample_n_features(num_sample=1000,seed_num=12)
    X_train, X_test, y_train, y_test= MakeDataset.split_examples(X,y)
    # MakeDataset.drawScatterGraph(X_train,y_train)
    regression = LinearRegression(500, 0.001)
    regression.fit(X_train,y_train)
    y_predict, weight, bias = regression.predict(X_test)
    # MakeDataset.drawScatterGraphWithLine(X_train,y_train,r1,b,weight,bias)

    # X,y,rx,b = MakeDataset.make_sample_n_features(num_sample=10,seed_num=532,n_features=3)
    # print(X)
    # print(y.shape)
