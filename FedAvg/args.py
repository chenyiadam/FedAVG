# -*- coding:utf-8 -*-
"""
@File: args.py
"""

# argparse的用法见csdn的收藏夹，或者https://blog.csdn.net/qq_41762249/article/details/122244624
# --E 相当于关键词参数，如果没有--直接是E，就是位置参数
# type=int 传入参数的类型
# default=20 当没有参数传入时，默认值为20， help='***' 表示对该参数的解释为***

'''
number of rounds of training: 训练次数
number of communication rounds：通信回合数，即上传下载模型次数。
number of total clients：客户端总数
input dimension ：输入维度
learning rate ：学习率
sampling rate ：采样率
local batch size ： 本地批量大小
type of optimizer ： 优化器类型
--device：有GPU就用，不然就用CPU

weight_decay ：权值衰减
weight decay（权值衰减）的使用既不是为了提高你所说的收敛精确度也不是为了提高收敛速度，其最终目的是防止过拟合。在损失函数中，weight decay是放在正则项（regularization）前面的一个系数，正则项一般指示模型的复杂度，所以weight decay的作用是调节模型复杂度对损失函数的影响，若weight decay很大，则复杂的模型损失函数的值也就大。https://blog.csdn.net/xuxiatian/article/details/72771609

step size： 步长
gamma： 伽马参数
--clients： 10个客户端 Task1_W_Zone1、Task1_W_Zone2、Task1_W_Zone3...Task1_W_Zone10
'''

import argparse
import torch


def args_parser():
    parser = argparse.ArgumentParser() # 可选参数： description='描述程序内容' 通过命令行 python **.py--help 调用出

    parser.add_argument('--E', type=int, default=20, help='number of rounds of training')   
    parser.add_argument('--r', type=int, default=5, help='number of communication rounds')
    parser.add_argument('--K', type=int, default=10, help='number of total clients')
    parser.add_argument('--input_dim', type=int, default=28, help='input dimension')
    parser.add_argument('--lr', type=float, default=0.08, help='learning rate')
    parser.add_argument('--C', type=float, default=0.8, help='sampling rate')
    parser.add_argument('--B', type=int, default=50, help='local batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay')
    parser.add_argument('--step_size', type=int, default=10, help='step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    
    clients = ['Task1_W_Zone' + str(i) for i in range(1, 11)]
    parser.add_argument('--clients', default=clients)

    # args = parser.parse_args()
    # args,unknow = parser.parse_known_args()
    
    args = parser.parse_known_args()[0]
    return args

