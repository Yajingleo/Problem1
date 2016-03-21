import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy import optimize

def Huber(a, b, x, y):
        if abs(y-a*x-b)<10:
            return (y-a*x-b)**2
        else:
            return 20*abs(y-a*x-b)-100


class nonGaussian_linear_model:
    a_L1=b_L1=a_L2=b_L2=a_M=b_M=None
    x=y=None
    
    # A constructor that reads the csv file from the directory 'file_directory'.
    def __init__(self, file_directory):
        with open(file_directory, 'rb') as f:
            reader = csv.reader(f)
            data = np.array(list(reader))
        self.x=[float(data[i][0]) for i in range(1,data.shape[0])]
        self.y=[float(data[i][1]) for i in range(1,data.shape[0])]
        self.mean=sum(self.y)/len(self.y)

    #A plot function to visualize the data. 
    def PlotXY(self):
        plt.scatter(self.x,self.y)
        plt.show()

    #L1LossFunction for A=[a,b].
    def L1Loss(self, A):
        return sum([abs(self.y[i]-A[0]*self.x[i]-A[1]) for i in range(len(self.x))])

    #L2LossFunction for A=[a,b].
    def L2Loss(self, A):
        return sum([(self.y[i]-A[0]*self.x[i]-A[1])*(self.y[i]-A[0]*self.x[i]-A[1]) for i in range(len(self.x))])


    #HuberLossFunction
    def HuberLoss(self, A):
        return sum([Huber(A[0],A[1],self.x[i],self.y[i]) for i in range(len(self.x))])
    
    #Linear regression using L1 loss function.
    def L1LossRegression(self):
        [self.a_L1, self.b_L1]=optimize.fmin(self.L1Loss, [2,2])
        self.residual_L1=np.array([self.y[i]-self.x[i]*self.a_L1-self.b_L1 for i in range(len(self.x))])
 
        print "The linear regression using L1 loss gives the coefficients (a,b)=(", self.a_L1,",", self.b_L1,")."

    #Gausssian linear regression using L2 loss.
    def L2LossRegression(self):
        [self.a_L2, self.b_L2]=optimize.fmin_cg(self.L2Loss, [2,2])
        self.residual_L2=np.array([self.y[i]-self.x[i]*self.a_L2-self.b_L2 for i in range(len(self.x))])
 
        print "The linear regression using L2 loss gives the coefficients (a,b)=(", self.a_L2,",", self.b_L2,")."

    #M-regression
    def MRegression(self):
        [self.a_M, self.b_M]=optimize.fmin_cg(self.HuberLoss, [2,2])
        self.residual_M=np.array([self.y[i]-self.x[i]*self.a_M-self.b_M for i in range(len(self.x))])
 
        print "The linear regression using L2 loss gives the coefficients (a,b)=(", self.a_M,",", self.b_M,")."

        
    #Plot the regression.
    def PlotRegression(self):
        if self.a_L1==None:
            self.L1LossRegression()
        if self.a_L2==None:
            self.L2LossRegression()
        if self.a_M==None:
            self.MRegression()
        
        X=np.linspace(min(self.x)-2, max(self.x)+2, 256)
        Y_L1=self.a_L1*X+self.b_L1
        Y_L2=self.a_L2*X+self.b_L2
        Y_M=self.a_M*X+self.b_M
        
        plt.plot(X, Y_L1, c="red", label="L1 Loss Regression")
        plt.plot(X, Y_L2, c="green", label="Least square Regression")
        plt.plot(X, Y_M, c="purple", label="M-Regression")
        plt.scatter(self.x, self.y)
        plt.ylabel('Values of y')
        plt.xlabel('Values of x')
        plt.legend()

        plt.show()

            
            
    #Plot the residual.
    def PlotResidual(self):
        if self.a_L1==None:
            self.L1LossRegression()
        if self.a_L2==None:
            self.L2LossRegression()

        self.residual_L1.sort()
        self.residual_L2.sort()

        X=np.arange(len(self.x))
        plt.bar(X,np.array(self.residual_L1), facecolor='#9999ff', label="L1 Loss")
        plt.bar(X,-np.array(self.residual_L2), facecolor='#ff9999', label="L2 Loss")
        plt.ylabel('Residuals')
        plt.legend()

        plt.show()    

    #Compute the R-squared
    def RSquared_L1(self):
        if self.a_L1==None:
            self.L1LossRegression()
                
        SSE=sum([(self.residual_L1[i])**2 for i in range(len(self.x))])
        SST=sum([(self.y[i]-self.mean)**2 for i in range(len(self.y))])
        return 1-SSE/SST

    def RSquared_M(self):
        if self.a_M==None:
            self.MRegression()
        SSE=sum([(self.residual_M[i])**2  for i in range(len(self.x))])
        SST=sum([(self.y[i]-self.mean)**2 for i in range(len(self.y))])
        return 1-SSE/SST

           

    #Plot the residual.
    def ResidualPDF(self):
        if self.a_L1==None:
            self.L1LossRegression()
        if self.a_L2==None:
            self.L2LossRegression()
            
        self.noise_L1.sort()
        self.noise_L2.sort()
        
        X=np.linspace(-40, 30, 100)
        Y_L1=Y_L2=[0 for i in range(100)]
        
        j, k=0,0
        for i in range(100):
            last_j, last_k = j, k
            while (j<len(self.noise_L1) and self.noise_L1[j]<X[i]):
                j+=1
            Y_L1[i]=j-last_j
            while (k<len(self.noise_L2) and self.noise_L2[k]<X[i]):
                k+=1
            Y_L2[i]=k-last_k
        
        Y_L1=np.array(Y_L1)
        Y_L2=np.array(Y_L2)
        plt.bar(X, +Y_L1, facecolor='#9999ff')
        plt.bar(X, -Y_L2, facecolor='#ff9999')
        
        
        
