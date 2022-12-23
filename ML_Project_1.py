
#We try to model the Indian economy using Linear Regression. The Cobb-Douglas production function gives Y=A K^(a) L^(1-a).
#Here Y is GDP, L is labour input, K is capital stock. A and a have to be found. A is also known as Total Factor Productivity or TFP.
#Thus, taking log both sides and simplifying we get ln(Y/L)=ln(A)+a ln(K/L). Thus, since ln(Y/L) and ln(K/L) are known data, we can find
#a and lnA using linear regression.
import numpy as np
import matplotlib.pyplot as plt

#Model class implementing linear regression.
class Model:
    def __init__(self,X,Y, alpha, lambda_):
        self.X,self.Y=self.input_transformation(X,Y)
        self.X,self.xmean,self.xsd=self.feature_scaling(self.X)
        self.Y,self.ymean,self.ysd=self.feature_scaling(self.Y)
        self.w,self.b=self.gradient_descent(self.X,self.Y,alpha,lambda_)
    def input_transformation(self,X,Y):
        Y=np.log(Y)
        K=np.zeros(len(X))
        L=np.zeros(len(X))
        for i in range(len(X)):
            K[i]=X[i][1]
            L[i]=X[i][0]
        K=np.log(K)
        L=np.log(L)
        return K-L, Y-L
 
#feature scaling using Z-score normalization to facilitate more efficient gradient descent.
    
    def feature_scaling(self,L):
        M=np.zeros(len(L))
        mean=sum(L)/len(L)
        
        sd=0
        for i in range(len(L)):
            sd+=(L[i]-mean)**2
        sd=np.sqrt(sd/len(L))
        for i in range(len(L)):
            M[i]=(L[i]-mean)/sd
        return M,mean,sd

#Mean-square cost function implementation, with regularization.
    def J(self,w,b,X,Y, lambda_):
        cost=0
        for i in range(len(X)):
            cost+=(w*X[i]+b-Y[i])**2
        cost/=2*len(X)
    #Regularization
        cost+=lambda_*(w**2)
        return cost
#Evaluation of partial derivatives, to be used in gradient descent algorithm.
    def partial_diff(self,ws,bs,X,Y, lambda_):
        dj_dw=0
        dj_db=0
        m=len(X)
        for i in range(len(X)):
            dj_dw+=(ws*X[i]+bs-Y[i])*X[i]
            dj_db+=(ws*X[i]+bs-Y[i])
        dj_dw/=m
        dj_dw+=2*ws*lambda_
        dj_db/=m
        return dj_dw,dj_db
#Gradient descent.
    def gradient_descent(self,X,Y,alpha,lambda_):
        ws=0
        bs=0
        
        for i in range(100000):
            dj_dw,dj_db=self.partial_diff(ws,bs,X,Y,lambda_)
            ws,bs=ws-alpha*dj_dw, bs-alpha*dj_db
        
        return ws,bs
    #User Interface functions
    def plot_model(self):
        plt.plot(self.X, self.Y, 'o')
        plt.plot(self.X, obj.w*self.X+obj.b)
        plt.show()
    def make_prediction(self,K,L):
        gdp=np.exp(self.ymean+self.b*self.ysd-self.w*self.xmean*(self.ysd/self.xsd))*K**(self.w*(self.ysd/self.xsd))*(L**(1-(self.w*(self.ysd/self.xsd))))
        print("Predicted GDP at given values of Capital Stock(K in Rs Crore) and Labour (in 1000s) is:", gdp)
    def print_TFP(self):
        print("TFP value obtained over given data is:",np.exp(self.ymean+self.b*self.ysd-self.w*self.xmean*(self.ysd/self.xsd)))
    def print_alpha(self):
        print("Value of capital share(alpha) over given data is:", self.w*(self.ysd/self.xsd))

#_main_
#Loading raw input data from https://www.rbi.org.in/Scripts/KLEMS.aspx into numpy arrays.
#X_train1 has (L,K) pairs for every yr from 1980-89, Y_train1 contains GDP values of corresponding yrs.
#Similarly X_train2 has (L,K) data for the next decade, and so on.
#X_train and Y_train contain (L,K) and GDP data over 1980-2020, a concatenation of the 4 previous arrays for overall analysis.
'''X_train1=np.array([[283562,4589560],[290516,4771087],[297754,4957352],[304367,5159054],[308533,5369912],[313020,5585743],[317881,5828080],[323182,6093647],
                 [330291,6379517],[338167,6701041]]
                 )
Y_train1=np.array([1413932, 1495736, 1542855, 1673434, 1739479, 1808887, 1886876, 1954569, 2170751, 2302276])




X_train2=np.array([[346374,7082957],[354928,7398038],[363845,7759510],[373144,8110730],[378332,8492327],[382284,8931868],[386507,9384459],[391017,9876592 ],
                   [395829,10439957],[400960,11116532 ]
                  ])
Y_train2=np.array([2439078, 2484241, 2607421, 2756885, 2924349, 3115811, 3363494, 3475424, 3690863, 3925152 ])




X_train3=np.array([[409770,11753033],[419968,12580648],[430803,13380059],[442325,14230827],[454591,15300930],[457649,16441610],[457950,17801847],
                  [458993,19407072],[460823,21105725],[463491,22831369]
                 ]) 
Y_train3=np.array([4076761,4286815,4423586,4766068,5087003,5507941,5950681,6388865,6669515,7127367])



X_train4=np.array([[467060,24645170],[471602,26664315],[471524,28739034],[470234,30764340],[469554,32776819],[469475,34912283],[469997,37239579],
                  [471123,39750222],[483343,42592391],[521855,45354176]
                  ])
Y_train4=np.array([7701398, 8106948, 8546276, 9063651, 9712129, 10491870, 11328285, 12034171, 12733798, 13219476]) 
 
 

X_train=np.concatenate((X_train1, X_train2, X_train3, X_train4), axis=0)
Y_train=np.concatenate((Y_train1, Y_train2, Y_train3, Y_train4), axis=0)

'''

