import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy import optimize

#load the data.
with open('/Users/yajingleo/Downloads/Problem1/data_1_5.csv', 'rb') as f:
            reader = csv.reader(f)
            data = np.array(list(reader))
            
#transform the data from string to float.
x=[float(data[i][0]) for i in range(1,data.shape[0])]
y=[float(data[i][1]) for i in range(1,data.shape[0])]


#Print the size, range
size=len(x)
print "range of x: [", min(x), ",",max(x),"]"
print "range of y: [", min(y), ",", max(y),"]"

#Plot the data.
plt.scatter(x,y)
plt.show()

#Define the loss function.
def L1Loss(A):
        return sum([abs(y[i]-A[0]*x[i]-A[1]) for i in range(len(x))])

def L2Loss(A):
        return sum([(y[i]-A[0]*x[i]-A[1])**2 for i in range(len(x))])

def Huber(a, b, x, y):
        if abs(y-a*x-b)<10:
            return (y-a*x-b)**2
        else:
            return 20*abs(y-a*x-b)-100

def HuberLoss(A):
        return sum([Huber(A[0],A[1],x[i],y[i]) for i in range(len(x))])

#Optimize the loss.
#L1 loss
[a_L1, b_L1]=optimize.fmin(L1Loss, [2,2])

#L2 loss
[a_L2, b_L2]=optimize.fmin_cg(L2Loss, [2,2])

#Huber loss
[a_M, b_M]=optimize.fmin_cg(HuberLoss, [2,2])


#Compute the residual.
residual_L1=np.array([y[i]-x[i]*a_L1-b_L1 for i in range(len(x))])
residual_L2=np.array([y[i]-x[i]*a_L2-b_L2 for i in range(len(x))])
residual_M=np.array([y[i]-x[i]*a_M-b_M for i in range(len(x))])
 
#Plot the regression results.
X=np.linspace(min(x)-2, max(x)+2, 256)
Y_L1=a_L1*X+b_L1
Y_L2=a_L2*X+b_L2
Y_M=a_M*X+b_M
        
plt.plot(X, Y_L1, c="red", label="L1 Loss Regression")
plt.plot(X, Y_L2, c="green", label="Least square Regression")
plt.plot(X, Y_M, c="purple", label="M-Regression")
plt.scatter(x, y)
plt.ylabel('Values of y')
plt.xlabel('The data has size of '+str(size) )
plt.legend()
plt.show()

#Plot the residual.

plt.subplot(1,3,1)
#residual_L1.sort()
X=np.arange(len(x))
plt.bar(X,np.array(residual_L1), facecolor='#9999ff', label="The residuals for L1 Loss model")
plt.legend()

plt.subplot(1,3,2)
#residual_L2.sort()
X=np.arange(len(x))
plt.bar(X,np.array(residual_L2), facecolor='#9999ff', label="The residuals for L2 Loss model")
plt.legend()

plt.subplot(1,3,3)
#residual_M.sort()
X=np.arange(len(x))
plt.bar(X,np.array(residual_M), facecolor='#9999ff', label="The residuals for M-regression")
plt.legend()
plt.show()

#Print the residuals.
print "The sum of absolute values of residuals for L1 is", sum(abs(residual_L1[i]) for i in range(len(x)))
print "The sum of absolute values of residuals for L2 is", sum(abs(residual_L2[i]) for i in range(len(x)))
print "The sum of absolute values of residuals for M is", sum(abs(residual_M[i]) for i in range(len(x)))

#Assess the parameters.
mean=sum(y)/len(y)

SSE_L1=sum([(a_L1*x[i]+b_L1-mean)**2 for i in range(len(x))])
SSE_L2=sum([(a_L2*x[i]+b_L2-mean)**2 for i in range(len(x))])
SSE_M=sum([(a_M*x[i]+b_M-mean)**2   for i in range(len(x))])
SST=sum([(y[i]-mean)**2         for i in range(len(y))])


print "The R^2 for L1 Loss model is ", SSE_L1/SST 
print "The R^2 for L2 Loss model is ", SSE_L2/SST 
print "The R^2 for M-regression model is ", SSE_M/SST 

#Pull out the large noise.
index_L2=[]
for i in range(len(x)):
    if (abs(y[i]-a_L2*x[i]-b_L2)<9):
        index_L2.append(i)
x_clean_L2=np.take(x,index_L2)
y_clean_L2=np.take(y,index_L2)


index_L1=[]
for i in range(len(x)):
    if (abs(y[i]-a_L1*x[i]-b_L1)<7):
        index_L1.append(i)
print index_L1
x_clean_L1=np.take(x,index_L1)
y_clean_L1=np.take(y,index_L1)


def L1Loss_clean(A):
    return sum([abs(y_clean_L1[i]-A[0]*x_clean_L1[i]-A[1]) for i in range(len(x_clean_L1))])


def L2Loss_clean(A):
    return sum([(y_clean_L1[i]-A[0]*x_clean_L1[i]-A[1])**2 for i in range(len(x_clean_L1))])

[a_L2_clean, b_L2_clean]=optimize.fmin_cg(L2Loss_clean, [2,2])


[a_L1_clean, b_L1_clean]=optimize.fmin(L1Loss_clean, [2,2])


#Plot the L1 regression after cleaning the noise.
X=np.linspace(min(x_clean_L1)-2, max(x_clean_L1)+2, 256)
Y_L1=a_L1*X+b_L1
Y_L1_clean=a_L1_clean*X+b_L1_clean
plt.plot(X, Y_L1, c="red", label="L1 Loss Regression")
plt.plot(X, Y_L1_clean, c="purple", label="Cleaned L1 Regression")
plt.scatter(x_clean_L1, y_clean_L1)
plt.xlabel('The data has size of '+str(len(x_clean_L1)) )
plt.legend()
plt.show()

#Plot the L2 regression after cleaning the noise. 
X=np.linspace(min(x_clean_L1)-2, max(x_clean_L1)+2, 256)
Y_L2_clean=a_L2_clean*X+b_L2_clean
Y_L2=a_L2*X+b_L2
        

plt.plot(X, Y_L2, c="green", label="Least square Regression")
plt.plot(X, Y_L2_clean, c="black", label="Cleaned L2 Regression")
plt.scatter(x_clean_L1, y_clean_L1)
plt.ylabel('Values of y')
plt.xlabel('The data has size of '+str(len(x_clean_L1)) )
plt.legend()
plt.show()

#Compute the change of the regression to assess the robustness.
print "L1",((a_L1-a_L1_clean)**2+(b_L1-b_L1_clean)**2)
print "L2",((a_L2-a_L2_clean)**2+(b_L2-b_L2_clean)**2)
