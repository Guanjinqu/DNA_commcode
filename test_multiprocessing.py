from multiprocessing import Process, Queue
import time
import itertools
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
import csv
from test_comm import Communication
from pyldpc import make_ldpc, encode, decode, get_message
from commpy.channelcoding import conv_encode
from commpy.utilities import dec2bitarray
import copy
import commpy.channelcoding.convcode as cc
from numpy import NaN, append, array, zeros, exp, log, empty
class MyProcess(Process):
    def __init__(self, class_dict_list,name_nums):
        Process.__init__(self)
        self.class_dict_list = class_dict_list
        self.name_nums = name_nums

    def output_result(self):
        count_nums = 0
        sum_nums = len(self.class_dict_list)
        for class_dict in self.class_dict_list :
            count_nums += 1
            C = Communication(
            mode = class_dict["mode"],
            H_row = class_dict["H_row"],
            G_row = class_dict["G_row"], 
            H_column = class_dict["H_column"],
            G_column = class_dict["G_column"],
            trellis_row = class_dict["trellis_row"],
            trellis_column = class_dict["trellis_column"],
            block_nums = class_dict["block_nums"],
            len_msg = class_dict["len_msg"],
            line_nums = class_dict["line_nums"],
            D = class_dict["D"],
            multiple_nums = class_dict["multiple_nums"],
            pr_dict =class_dict["pr_dict"],
            iterations = class_dict["iterations"]
            )
            out = C.fast_mode()
            class_dict["result"] = out
            f = open("test_result_620.txt","a+")
            f.write(str([class_dict["mode"],class_dict["multiple_nums"],class_dict["multiple_nums"],class_dict["iterations"],out]))
            f.write('\n')
            print(self.name_nums,count_nums/sum_nums,count_nums,sum_nums)


    def run(self): 
        self.output_result()


if __name__ == '__main__':
    # 创建进程数
    PROCESS_nums = 3

    base_dict = {
        "H_row" : None,
        "G_row" : None, 
        "H_column" : None,
        "G_column" : None,
        "trellis_row" : None,
        "trellis_column" : None,
        "block_nums" : 6,
        "len_msg" : 30,
        "line_nums" : 30,
        "D" : 1,
        "multiple_nums" : 5,
        "pr_dict" :{
        "ps":0.02,
        "pd":0.0001,
        "pi":0.0001,
        "column":0},
        "iterations" : 3
        
    }
    mode_list = ["Interleaver_1","Interleaver_2","Interleaver_3"]
    pr_dict ={
        "ps":0.05,
        "pd":0.025,
        "pi":0.025,
        "column":0}
    ldpc_list = [[60,2,6]]

    
    data_dict = {}
    for i in range(PROCESS_nums):
        data_dict[i] = []

    memory1 = array([1])                     #寄存器的个数

    g_matrix1 = array([[1,2]])               #生成矩阵的样子

    fd1 = array([[3]])

    trellis1 = cc.Trellis(memory1,g_matrix1,feedback=fd1,code_type='rsc')
    index_process = 0
    for ldpc_p in ldpc_list:
        H, G = make_ldpc(ldpc_p[0], ldpc_p[1], ldpc_p[2], systematic=True, sparse=False,seed=1)
        for mode in mode_list:
                    for iterations in range(1,10):
                        if mode == "Interleaver_1":
                            now_dict = copy.deepcopy(base_dict)
                            now_dict["mode"] = "Interleaver_1"
                            now_dict["trellis_row"] = trellis1
                            now_dict["trellis_column"] = trellis1
                            now_dict["pr_dict"] = pr_dict
                            now_dict["multiple_nums"] = 5
                            now_dict["iterations"] = iterations
                            data_dict[0].append(now_dict) 
            
                        if mode == "Interleaver_2":
                            now_dict = copy.deepcopy(base_dict)
                            now_dict["mode"] = "Interleaver_2"
                            now_dict["trellis_row"] = trellis1
                            now_dict["H_column"] = H
                            now_dict["G_column"] = G
                            now_dict["pr_dict"] = pr_dict
                            now_dict["multiple_nums"] = 5
                            now_dict["iterations"] = iterations
                            data_dict[1].append(now_dict) 

                        if mode == "Interleaver_3":
                            now_dict = copy.deepcopy(base_dict)
                            now_dict["mode"] = "Interleaver_3"
                            now_dict["H_row"] = H
                            now_dict["G_row"] = G
                            now_dict["H_column"] = H
                            now_dict["G_column"] = G
                            now_dict["pr_dict"] = pr_dict
                            now_dict["multiple_nums"] =5
                            now_dict["iterations"] = iterations
                            data_dict[2].append(now_dict)                        


    #q_output = Queue(N_PROCESS*2)
    # Init all processes

    process_dict = {}
    for i in range(PROCESS_nums):
        process_dict[i] = MyProcess(data_dict[i],i)
    # Start all processes
    for i in process_dict:
        process_dict[i].start()
    # Get output
    for i in process_dict:
        process_dict[i].join()