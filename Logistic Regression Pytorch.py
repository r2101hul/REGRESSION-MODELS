# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 19:35:19 2018

@author: Rahul
"""
import torch
from torch.autograd import Variable
import torch.nn.functional as F

x_data=Variable(torch.Tensor([[1.0],[2.0],[3.0],[4.0]]))
y_data=Variable(torch.Tensor([[0.],[0.],[1.],[1.]]))


class LOG(torch.nn.Module):
    def __init__(self):
        super(LOG, self).__init__()
        self.linear=torch.nn.Linear(1,1)
    def forward(self,x):
        y_pred=F.sigmoid(self.linear(x))
        return y_pred
    
model=LOG()
criterion=torch.nn.BCELoss(size_average=True)
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)


for epoch in range(1000):
    y_pred=model(x_data) 
    loss=criterion(y_pred,y_data)
    print(epoch,loss.data[0])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

var=Variable(torch.Tensor([[1.0]]))
print("predict 1 ",1.0,model(var).data[0][0]>0.5)


var=Variable(torch.Tensor([[7.0]]))
print("predict 7 ",7.0,model(var).data[0][0]>0.5)





    
    
    
        