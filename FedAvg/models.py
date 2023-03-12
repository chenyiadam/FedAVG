# -*- coding:utf-8 -*-
"""
@File: models.py
"""
import numpy as np
from torch import nn


class BP:
    def __init__(self, args, file_name):
        self.file_name = file_name
        self.len = 0
        self.args = args
        self.input = np.zeros((args.B, args.input_dim))  # self.B samples per round  （本地批量大小=50，输入维度=28）
        
        self.w1 = 2 * np.random.random((args.input_dim, 20)) - 1  # limit to (-1, 1) （28，20）
        self.z1 = 2 * np.random.random((args.B, 20)) - 1  #np.random.random生成args.B=50行 20列的0-1浮点数；*2→（0-2），再-1，变成（-1，1）
        self.hidden_layer_1 = np.zeros((args.B, 20))     #（50，20）
        
        self.w2 = 2 * np.random.random((20, 20)) - 1     #（20，20）
        self.z2 = 2 * np.random.random((args.B, 20)) - 1  #（50，20）
        self.hidden_layer_2 = np.zeros((args.B, 20))     #（50，20）
        
        self.w3 = 2 * np.random.random((20, 20)) - 1     #（20，20）
        self.z3 = 2 * np.random.random((args.B, 20)) - 1  #（50，20）
        self.hidden_layer_3 = np.zeros((args.B, 20))     #（50，20）
        
        self.w4 = 2 * np.random.random((20, 1)) - 1     #（20，1）
        self.z4 = 2 * np.random.random((args.B, 1)) - 1  #（50，1）
        self.output_layer = np.zeros((args.B, 1))      #（50，1）
        
        self.loss = np.zeros((args.B, 1))           #（50，1）

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deri(self, x):
        return x * (1 - x)

    def forward_prop(self, data, label):
        self.input = data
        # self.input（50，28）  self.w1（28， 20）  self.z1（50， 20）
        self.z1 = np.dot(self.input, self.w1) # np.dot 计算过程就是将向量中对应元素相乘，再相加所得。即普通的向量乘法运算。
        
        self.hidden_layer_1 = self.sigmoid(self.z1) # self.hidden_layer_1（50， 20）
        
        self.z2 = np.dot(self.hidden_layer_1, self.w2)  #self.w2（20，20） self.z2（50， 20）
        self.hidden_layer_2 = self.sigmoid(self.z2) # self.hidden_layer_2（50， 20）
        
        self.z3 = np.dot(self.hidden_layer_2, self.w3)  #self.w3（20，20） self.z3（50， 20）
        self.hidden_layer_3 = self.sigmoid(self.z3)    #（50，20）
        
        self.z4 = np.dot(self.hidden_layer_3, self.w4)  #self.w4 （20，1） self.z4（50，1）
        self.output_layer = self.sigmoid(self.z4)     #self.output_layer（50，1）
        # error

        self.loss = 1 / 2 * (label - self.output_layer) ** 2  ##（50，1）  why 1/2 ?

        return self.output_layer

    def backward_prop(self, label):
        # w4
        l_deri_out = self.output_layer - label
        l_deri_z4 = l_deri_out * self.sigmoid_deri(self.output_layer)
        l_deri_w4 = np.dot(self.hidden_layer_3.T, l_deri_z4)
        # w3
        l_deri_h3 = np.dot(l_deri_z4, self.w4.T)
        l_deri_z3 = l_deri_h3 * self.sigmoid_deri(self.hidden_layer_3)
        l_deri_w3 = np.dot(self.hidden_layer_2.T, l_deri_z3)
        # w2
        l_deri_h2 = np.dot(l_deri_z3, self.w3.T)
        l_deri_z2 = l_deri_h2 * self.sigmoid_deri(self.hidden_layer_2)
        l_deri_w2 = np.dot(self.hidden_layer_1.T, l_deri_z2)
        # w1
        l_deri_h1 = np.dot(l_deri_z2, self.w2.T)
        l_deri_z1 = l_deri_h1 * self.sigmoid_deri(self.hidden_layer_1)
        l_deri_w1 = np.dot(self.input.T, l_deri_z1)
        # update
        self.w4 -= self.args.lr * l_deri_w4  # self.args.lr 学习率=0.08  实则梯度下降
        self.w3 -= self.args.lr * l_deri_w3
        self.w2 -= self.args.lr * l_deri_w2
        self.w1 -= self.args.lr * l_deri_w1

'''
用到nn.Model和nn.Parameter来完成一个更加清晰简洁的训练循环。我们继承nn.Module(它是一个能够跟踪状态的类)。在这个例子中，我们想要新建一个类，实现存储权重，偏置和前向传播步骤中所有用到方法。nn.Module包含了许多属性和方法（比如.parameters()和.zero_grad()），我们会在后面用到。

nn.Module是一个PyTorch中特有的概念，它是一个会经常用到的类。
'''

class ANN(nn.Module):
    def __init__(self, args, name):
        super(ANN, self).__init__()  
        #  super().__init__()，就是继承父类的__init__()方法，同样可以使用super()去继承其他方法
        # 目的是引入父类的初始化方法给子类进行初始化！
        # super 方法 https://blog.csdn.net/zhulewen/article/details/125830877
        
        self.name = name
        self.len = 0
        self.loss = 0
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(args.input_dim, 16) #（28，16）
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, data):
        x = self.fc1(data)
        x = self.sigmoid(x)
        
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        x = self.fc3(x)
        x = self.sigmoid(x)
        
        x = self.fc4(x)
        x = self.sigmoid(x)

        return x
