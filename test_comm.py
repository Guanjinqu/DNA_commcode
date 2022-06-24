import matplotlib as plt
import numpy as np
import pandas as pd
import random
from sympy import true
from platform import java_ver
import sys
from re import L, X
import copy
import numpy as np
from numpy import NaN, array, zeros, exp, log, empty
import math
from commpy.channelcoding import conv_encode
from commpy.utilities import dec2bitarray
import time
import commpy.channelcoding.convcode as cc
import random
random.seed(1)

from pyldpc import make_ldpc, encode, decode, get_message
random.seed(1)
from test_encode import conv_encode,LDPC_encode,Interleaver_1_encode,Interleaver_2_encode,Interleaver_3_encode
from test_channel import Channel_conv,Channel_LDPC,Channel_Interleaver
from test_decode import Decode_conv,Decode_LDPC,Decode_Interleaver_1,Decode_Interleaver_2,Decode_Interleaver_3
from test_acc import Acc_conv,Acc_LDPC,Acc_I1,Acc_I2,Acc_I3


class Communication :

    def __init__(

        self,
        mode,
        H_row = None,
        G_row = None, 
        H_column = None,
        G_column = None,
        trellis_row = None,
        trellis_column = None,
        block_nums = 10,
        len_msg = 100,
        line_nums = 100,
        D = 2,
        multiple_nums = 5,
        pr_dict ={
        "ps":0.02,
        "pd":0.0001,
        "pi":0.0001,
        "column":0},
        iterations = 3
        ) :
        self.block_nums = block_nums
        self.len_msg = len_msg
        self.line_nums = line_nums
        self.D = D
        self.pr_dict = pr_dict
        self.multiple_nums = multiple_nums
        self.iterations = iterations
        self.mode = mode
        if mode == "conv" :
            self.trellis_row = trellis_row
        elif mode == "LDPC" :
            self.H_row = H_row
            self.G_row = G_row
            self.len_msg = int(G_row.shape[1])
        elif mode == "Interleaver_1" :
            self.trellis_row = trellis_row
            self.trellis_column = trellis_column
        elif mode == "Interleaver_2" :
            self.line_nums = int(G_column.shape[1])
            self.trellis_row = trellis_row
            self.H_column = H_column
            self.G_column = G_column
        elif mode == "Interleaver_3" :
            self.line_nums = int(G_column.shape[1])
            self.len_msg = int(G_row.shape[1])
            self.H_row = H_row
            self.G_row = G_row            
            self.H_column = H_column
            self.G_column = G_column


    def fast_mode(self):
        if self.mode == "conv" :
            encode_list = conv_encode(self.len_msg,self.line_nums,self.trellis_row,input_mode = False)
            print("encode ok")
            msg_list = encode_list[0]
            code_list = encode_list[1]
            channel_code_list = Channel_conv(code_list,self.pr_dict,self.multiple_nums)
            print("channel ok")
            decode_code_list = Decode_conv(channel_code_list,self.pr_dict,self.trellis_row,self.len_msg,self.D,self.multiple_nums,self.line_nums)
            print("decode ok")
            return Acc_conv(msg_list,decode_code_list)

        elif self.mode == "LDPC" :
            encode_list = LDPC_encode(self.len_msg,self.line_nums,self.G_row,input_mode = False)
            print("encode ok")
            msg_list = encode_list[0]
            code_list = encode_list[1]
            channel_code_list = Channel_LDPC(code_list,self.pr_dict,self.multiple_nums)
            print("channel ok")
            decode_code_list = Decode_LDPC(channel_code_list,self.pr_dict,self.H_row,self.G_row,self.D,self.multiple_nums,self.line_nums)
            print("decode ok")
            #print(msg_list,decode_code_list)
            return Acc_LDPC(msg_list,decode_code_list)

        elif self.mode == "Interleaver_1" :
            #print(self.len_msg)
            encode_list = Interleaver_1_encode(self.len_msg,self.line_nums,self.block_nums,self.trellis_row,self.trellis_column,input_mode = False)
            #print("encode ok")
            
            msg_list = encode_list[0]
            code_list = encode_list[1] 
            channel_code_list  =  Channel_Interleaver(code_list,self.pr_dict,self.multiple_nums)  
            print("channel ok")
            #time.sleep(20)
            #print(msg_list)
            decode_code_list = Decode_Interleaver_1(self.len_msg,self.line_nums,channel_code_list,self.pr_dict,self.trellis_row,self.trellis_column,self.block_nums,self.iterations,self.D)
            #print(msg_list,decode_code_list)
            #print(np.shape(msg_list),np.shape(decode_code_list))
            return Acc_I1(msg_list,decode_code_list)

        elif self.mode == "Interleaver_2" :
            encode_list = Interleaver_2_encode(self.len_msg,self.block_nums,self.trellis_row,self.G_column,input_mode = False)
            print("encode ok")
            msg_list = encode_list[0]
            code_list = encode_list[1]    
            channel_code_list = Channel_Interleaver(code_list,self.pr_dict,self.multiple_nums)
            print("channel ok")
            decode_code_list = Decode_Interleaver_2(self.len_msg,self.block_nums,channel_code_list,self.pr_dict,self.trellis_row,self.H_column,self.G_column,self.iterations,self.D)             
            print("decode ok")
            return Acc_I2(msg_list,decode_code_list)

        elif self.mode == "Interleaver_3":
            encode_list = Interleaver_3_encode(self.block_nums,self.G_row,self.G_column,input_mode = False)
            print("encode ok")
            msg_list = encode_list[0]
            code_list = encode_list[1]    
            channel_code_list = Channel_Interleaver(code_list,self.pr_dict,self.multiple_nums) 
            print("channel ok") 
            decode_code_list = Decode_Interleaver_3(self.block_nums,channel_code_list,self.pr_dict,self.H_row,self.G_row,self.H_column,self.G_column,self.D,self.iterations)             
            print("decode ok")
            return Acc_I3(msg_list,decode_code_list)





##################################################################

if __name__ == '__main__':

    #这个是测试单纯卷积码性能的

    memory1 = array([1])                     #寄存器的个数

    g_matrix1 = array([[1,2]])               #生成矩阵的样子

    fd1 = array([[3]])

    trellis1 = cc.Trellis(memory1,g_matrix1,feedback=fd1,code_type='rsc')


    memory1 = array([2])                     #寄存器的个数

    g_matrix1 = array([[1,5]])               #生成矩阵的样子

    fd1 = array([[7]])

    trellis2 = cc.Trellis(memory1,g_matrix1,feedback=fd1,code_type='rsc')

    memory1 = array([3])                     #寄存器的个数

    g_matrix1 = array([[1,0o17]])               #生成矩阵的样子

    fd1 = array([[13]])

    trellis3 = cc.Trellis(memory1,g_matrix1,feedback=fd1,code_type='rsc')

    trellis_list = [trellis1]

    pr_now_dict ={
        "ps":0.05,
        "pd":0.025,
        "pi":0.025,
        "column":0}
    pr_dict_list2 = [pr_now_dict]    
    pr_dict ={
        "ps":0.002,
        "pd":0.001,
        "pi":0.001,
        "column":0}
    pr_dict_list = [pr_dict]
    for i in range(20) :
        now_dict= copy.deepcopy(pr_dict)
        now_dict["ps"] += 0.005*i
        now_dict["pd"] += 0.0025*i
        now_dict["pi"] += 0.0025*i
        pr_dict_list.append(now_dict )
    i=0
    H, G = make_ldpc(60, 2, 6, systematic=True, sparse=False,seed=1)
    for trellis in trellis_list:
        
        for pr_dict in pr_dict_list:
            for multiple_nums in range(1,2):
                for iterations in range(1,2):
                    i=i+1

                    test =Communication(
                        mode = "conv",
                        trellis_row= trellis,
                        trellis_column=trellis,
                        line_nums=30,
                        pr_dict=pr_dict,
                        block_nums = 6,
                        len_msg =30,
                        D = 1,
                        multiple_nums =multiple_nums,
                        iterations = iterations ) 
                    fin = test.fast_mode()
                    
                    f = open("test1_result10.txt","a+")
                    f.write(str(["I1_p",trellis.total_memory,pr_dict,multiple_nums,iterations,fin ]))
                    f.write('\n')
                    print(i)
