import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 

x_train=np.array([3,4,5,6,7,])
y_train=np.array([7,8,9,10,11])

w=100
b=200

def compute_model_output(x,w,b):
    m=x.shape[0]
    f_wb= np.zeros(m)
    for i in range(m):
        f_wb[i]=w*x[i]+b

    return f_wb

tmp_f_wb= compute_model_output(x_train,w,b)

def compute_Cost(x,y,w,b):
    m=x.shape[0]
    cost_sum=0
    for i in range(m):
        f_wb=w*x[i]+b
        cost=(f_wb-y[i])**2
        cost_sum =cost_sum+cost
    total_cost= (1/(2*m))*cost_sum
    return total_cost

w=150
b=500

tmp_f_wb1=compute_Cost(x_train, y_train, w, b)
print(f'total cost is :{tmp_f_wb1}')

