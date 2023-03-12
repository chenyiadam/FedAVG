import numpy as np
import random
import copy
import sys

sys.path.append('../')
from algorithms.bp_nn import train, test
from models import BP 
from args import args_parser  # 一些传入参数，见args.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 只看error


#-----------------tf用于设置随机数
import tensorflow as tf

clients_wind = ['Task1_W_Zone' + str(i) for i in range(1, 11)]

# Implementation for FedAvg by numpy. 通过numpy实现FedAvg。
class FedAvg:
    def __init__(self, args): #self 默认必须参数，有类中全局变量之效，args表示，调用FedAvg时，必须传入的参数
        self.args = args
        self.clients = args.clients
        self.nn = BP(args=args, file_name='server') # BP是models中的一个类，同样需要传入参数。file_name方便后面为每个客户端取名
        self.nns = []
        # distribution
        for i in range(self.args.K): #args.K，客户端总数； 子程序为每一个客户端构造了一个BP类
            #copy.deepcopy() 深复制的用法是将某个变量的值赋给另一个变量(此时两个变量地址不同)，因为地址不同，所以变量间互不干扰
            s = copy.deepcopy(self.nn)  
            s.file_name = self.clients[i]
            self.nns.append(s)

    def server(self):
        for t in range(self.args.r): #通信回合数，即本地模型上传下载全局模型次数
            print('round', t + 1, ':') # 输出：round1、round2、round3、round4、round5
#             m = np.max([int(self.args.C * self.args.K), 1]) # 抽样率*客户端总数，即每一轮参与训练的客户端数量，至少有1个客户端参与
            m = 5
            print(m)
            # sampling
            index = random.sample(range(0, self.args.K), m) #在0-（k-1）之间共k个中抽取m个序号，注意是序号/索引
            print(len(index))
            # dispatch
            self.dispatch(index) # 下面定义了dispatch函数：抽中的m本地客户端从服务端下载4个参数
            # local updating
            self.client_update(index) # 下面定义了client_update函数：抽中的m个客户端进行本地训练
            # aggregation
            self.aggregation(index) # 下面定义了aggregation函数：抽中的m个客户端，上传本地训练结果参数

        # return global model
        return self.nn #返回最终聚合后的模型

    def aggregation(self, index):
        # update w
        s = 0 #用来计一轮抽中的m个本地客户端总的样本数
        for j in index:
            # normal
            s += self.nns[j].len
        w1 = np.zeros_like(self.nn.w1) #np.zeros_like：生成和self.nn.w1一样的零阵，下同
        w2 = np.zeros_like(self.nn.w2)
        w3 = np.zeros_like(self.nn.w3)
        w4 = np.zeros_like(self.nn.w4)
        
        #-----------------自增1018
        nois = 0.05
        for j in index: # 对上传的每一个本地模型进行权重的加权求和，权重为该客户端样本数/该轮中参与训练的总样本数
            # normal
            w1 += self.nns[j].w1 * (self.nns[j].len / s) + tf.random.normal([1],mean=0, stddev=nois).numpy()
            w2 += self.nns[j].w2 * (self.nns[j].len / s) + tf.random.normal([1],mean=0, stddev=nois).numpy()
            w3 += self.nns[j].w3 * (self.nns[j].len / s) + tf.random.normal([1],mean=0, stddev=nois).numpy()
            w4 += self.nns[j].w4 * (self.nns[j].len / s) + tf.random.normal([1],mean=0, stddev=nois).numpy()
        # update server 更新服务端参数
        self.nn.w1, self.nn.w2, self.nn.w3, self.nn.w4 = w1, w2, w3, w4

    def dispatch(self, index):
        # distribute
        for i in index:
            self.nns[i].w1, self.nns[i].w2, self.nns[i].w3, self.nns[i].w4 = self.nn.w1, self.nn.w2, self.nn.w3, self.nn.w4

    def client_update(self, index):  # update nn
        for k in index:
            self.nns[k] = train(self.args, self.nns[k])

    def global_test(self):
        model = self.nn #最终聚合后的模型
        c = clients_wind  # 10个客户端名称 Task1_W_Zone1、Task1_W_Zone2、Task1_W_Zone3...Task1_W_Zone10
        for client in c:
            print(client)
            model.file_name = client 
            test(self.args, model)

'''
L1损失函数: mae
均方根误差: rmse
https://blog.csdn.net/qq_45758854/article/details/125807544
'''

def main():
    args = args_parser()
    fed = FedAvg(args)
    fed.server()
    fed.global_test()


    if __name__ == '__main__':
    main()
