import numpy as np
import MakeDataset
from tqdm import tqdm
class LinearRegression:
    def __init__(self,iteration = 500,learning_rate=0.0001) :
        self.lr = learning_rate
        self.iter = iteration
        self.weight=None
        self.bias =0
    def fit(self,X,y):
        #i modify fit function "y = r1 * x1+ r2 * x2 + r3 * x3+ ... + b" is available.
        n_feature,length = X.shape
        self.weight = np.ones(n_feature)

        for _ in tqdm(range(self.iter)):
            y_pred =np.dot(self.weight,X)+ self.bias
            print(y_pred.shape)
            dw = (1 / length ) * np.dot((y_pred-y),X.T)
            db = (1 /length) * np.sum(y_pred-y)
            self.weight -= dw.reshape(-1) * self.lr
            self.bias -=db * self.lr

        print("weight : ",self.weight)
    def predict(self,X):
        return np.dot(self.weight,X)+ self.bias ,self.weight,self.bias





if __name__ == "__main__":
    print("main")
    X,y, r1 ,b = MakeDataset.make_sample_n_features(num_sample=1000,seed_num=12,n_features=5)
    print(r1,b)
    X_train, X_test, y_train, y_test= MakeDataset.split_examples(X,y)
    print("X_train shape : ",X_train.shape)
    print("y_train shape: ",y_train.shape)
    print("original weight :", r1)
    #MakeDataset.drawScatterGraph(X_train,y_train)
    regression = LinearRegression(1000, 0.0001)
    regression.fit(X_train,y_train)
    y_pred, weight, bias =regression.predict(X_test)



    # y_predict, weight, bias = regression.predict(X_test)
    #MakeDataset.drawScatterGraphWithLine(X_train.reshape(-1),y_train.reshape(-1),r1.reshape(-1),b,weight.reshape(-1),bias)

    # X,y,rx,b = MakeDataset.make_sample_n_features(num_sample=10,seed_num=532,n_features=3)
    # print(X)
    # print(y.shape)
