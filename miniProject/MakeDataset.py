import random
import numpy as np
import matplotlib.pyplot as plt
#

def measure_time(func):
    def wrapper(*args, **kwargs):
        import time 
        start = time.time()
        result=func(*args,**kwargs)
        end= time.time()
        print(f"function {func.__name__} took {end-start}.")
        return result
    return wrapper

def make_regression_sample(seed_num=1234, num_sample=500):
    '''make y= r1 * x + b samples 
    returns (X,y) , r1, b
    0 <= r1 < 10 and immutable
    0 <= bias < 10 and has a noise (-5 < x < 5)
       '''
    random.seed(seed_num)
    np.random.seed(seed_num)
    r1 = random.randint(0,10)
    b= random.uniform(0,10)

    np_noise = np.random.uniform(low=-5,high=5,size=num_sample)
    X = np.random.uniform(low=0, high=50,size=num_sample)
    y= (np.dot(r1,X) + b) + np_noise
    return X,y ,r1, b

@measure_time
def make_sample_n_features(seed_num=1234, num_sample=500,n_features=3):
    '''make y= r1 * x1 + r2 * x2 +r3 * x3 ... + rx * x + b samples 
    returns (X,y) , rx, b
    0 <= rx < 10 and immutable
    0 <= bias < 10 and has a noise (-5 < x < 5)
       '''
    np.random.seed(seed_num)
    
    rx = np.array([np.random.randint(0,10,size=n_features)])
    b= np.random.uniform(0,10)

    np_noise = np.random.uniform(low=-5,high=5,size=num_sample)
    X = np.random.uniform(low=0, high=50,size=num_sample *n_features).reshape(n_features,-1)
    print(X.shape,rx.shape)
    y= (np.dot(rx,X) + b) + np_noise
    return X,y ,rx, b

def split_examples(X,y,ratio=0.8,random_seed=124):
    '''
        Split given X,y by using given ratio.
        returns X_train, X_test, y_train,y_test
    '''
    length = int(len(X) * ratio) if X.ndim ==1 else int(len(X[0]) * ratio)
    arr = np.vstack((X,y)).transpose()
    n_feature = len(X)
    np.random.seed(random_seed)
    np.random.shuffle(arr)
    train_Arr , test_Arr = np.split(arr,(length,))
    train_Arr = train_Arr.transpose()
    test_Arr = test_Arr.transpose()
    
    return train_Arr[:n_feature], test_Arr[:n_feature], train_Arr[n_feature:] ,test_Arr[n_feature:]

def drawScatterGraph(X,y):
    plt.scatter(X,y)
    plt.title("scatter")
    plt.xlabel("X")
    plt.ylabel('y')
    plt.show()

def drawScatterGraphWithLine(X,y,r1,b,weight,bias):
    plt.scatter(X,y)
    x1 = list(range(50))
    y_orgin = [r1* x +b for x in x1] 
    y_pred = [weight * x +bias for x in x1]
    plt.plot(x1,y_orgin,color="r",label="origin")
    plt.plot(x1,y_pred,color="black",label ="predict" )

    plt.title("scatter")
    plt.xlabel("X")
    plt.ylabel('y')
    plt.legend(loc="upper left")
    plt.show()



if __name__ == "__main__":
    X,y, rx ,b = make_sample_n_features(seed_num=4, num_sample=5,n_features=3)
    xtrain,xtest,ytrain,ytest=split_examples(X,y)
    print(xtest)
    # print(split_examples(X,y))





