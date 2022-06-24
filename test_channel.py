
#本代码为信道模拟
from numpy import NaN, array, zeros, exp, log, empty
import numpy as np
import random

pr_dict={
        "ps":0.02,
        "pd":0.0001,
        "pi":0.0001,
        "column":0} 
pr_dict["pt"]=((1-pr_dict["pi"])*(1-pr_dict["pd"]))

def channel_model_unit(code,pr_dict):
    #信道模拟函数

    unit_list=[1,0]
    pr_dict["pt"]=((1-pr_dict["pi"])*(1-pr_dict["pd"]))
    new_pr_dict={}
    for key in pr_dict:
        new_pr_dict[key]=10000*pr_dict[key]


    af_code=[]
    if random.randint(1,10000) <= new_pr_dict["column"]:
        for i in range(code.size) :
            af_code.append(2)
    else:
        for i in range(code.size) :
            if random.randint(1,10000) <= new_pr_dict["pi"]:
                af_code.append(random.choice(unit_list))
            if random.randint(1,10000) <= new_pr_dict["pd"]:
                continue
            if random.randint(1,10000) <= new_pr_dict["ps"]:
                af_code.append(abs(1-code[i]))
            else:
                af_code.append(code[i])
    #print(len(af_code),list[2])
    af_code_array = np.array(af_code)
    return af_code_array

"""
#########################
在此我统一所有通过信道后的数据类型：
所有的数据放在一个list中，每条信息为list中的一个元素，且也是list格式
每条信息的格式标准为[块索引，行索引，码字]
对于非交织器的块索引固定为0
注意 非交织器下生成的码字格式为数组。
#########################
"""


def Channel_conv(code_metrics,pr_dict,multiple_nums):
    after_channel_code_list = []
    for i in range(code_metrics.shape[0]):
        bf_code = code_metrics[i]
        for j in range(multiple_nums):
            af_code = channel_model_unit(bf_code,pr_dict)
            after_channel_code_list.append([0,i,af_code])
    return after_channel_code_list

def Channel_LDPC(code_metrics,pr_dict,multiple_nums):
    after_channel_code_list = []
    for i in range(code_metrics.shape[0]):
        bf_code = code_metrics[i]
        for j in range(multiple_nums):
            af_code = channel_model_unit(bf_code,pr_dict)
            after_channel_code_list.append([0,i,af_code])
    return after_channel_code_list

def Channel_Interleaver(code_metrics_list,pr_dict,multiple_nums):
    after_channel_code_list = []
    for k in range(len(code_metrics_list)):
        code_metrics = code_metrics_list[k]
        for i in range(code_metrics.shape[0]):
            bf_code = code_metrics[i]
            for j in range(multiple_nums):
                af_code = channel_model_unit(bf_code,pr_dict)
                after_channel_code_list.append([k,i,af_code])     
    return after_channel_code_list
    

