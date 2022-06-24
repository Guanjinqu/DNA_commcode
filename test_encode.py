
#本代码为编码的全部构造


import sys
import numpy as np
from numpy import NaN, append, array, zeros, exp, log, empty
import math
from commpy.channelcoding import conv_encode
from commpy.utilities import dec2bitarray
import time
import commpy.channelcoding.convcode as cc
import random
#random.seed(1)

from pyldpc import make_ldpc, encode, decode, get_message



def conv_encode(len_line,line_nums,trellis,input_mode = False) :
    """
    单纯卷积码的编码器
    输入： 
    len_line :序列数
    line_nums :序列长度
    trellis : 卷积码结构
    input_mode :是否是输入了真实的信息

    输出：
    字典形式
    """
    k = int(trellis.k)
    n = int(trellis.n)
    msg_array = zeros([line_nums,len_line],dtype= int)
    for i in range(line_nums):
        for j in range(len_line-k*trellis.total_memory):
            msg_array[i,j] = int(random.choice([0,1]))
    msg_array =np.array(msg_array)
    #print(msg_array)
    code_array = zeros([line_nums,int(((len_line+trellis.total_memory)/k)*n)],dtype= int)
    for i in range(line_nums):
        code_array[i] = cc.conv_encode(msg_array[i],trellis, termination='term')
    
    
    return [msg_array,code_array]


def binaryproduct(X, Y):

    A = X.dot(Y)
    try:
        A = A.toarray()
    except AttributeError:
        pass
    return A % 2




def LDPC_encode(len_msg,line_nums,G,input_mode = False):
    v = G.shape[0]
    msg_array = zeros([line_nums,len_msg],dtype= int)
    code_array = zeros([line_nums,v],dtype= int)
    for i in range(line_nums):
        for j in range(len_msg):
            msg_array[i,j] = int(random.choice([0,1]))
        code_array[i] = binaryproduct(G, msg_array[i])

    return [msg_array,code_array]




def Interleaver_1_encode(len_msg,line_nums,block_nums,trellis_row,trellis_column,input_mode = False):
    print(len_msg)
    n_row = trellis_row.n
    k_row = trellis_row.k
    memory_row = trellis_row.total_memory

    n_column = trellis_column.n
    k_column = trellis_column.k
    memory_column = trellis_column.total_memory

    block_nums = int(block_nums)
    #总共流程有三个矩阵类，
    #第一个是2*line_nums - memory_column 乘 len_msg
    #第二个是（2*line_nums）/k*n 乘 len_msg (完成了纵向编码)
    #第三个是（2*line_nums）/k*n 乘 （len_msg+memory_row）/k*n  （完成了横向编码）

    #构造第一个矩阵
    msg_metrics_list = [np.random.randint(0,1,size = [line_nums,len_msg])]
    for i in range(int(block_nums/2)-1):
        msg_metrics_list.append(np.random.randint(0,2,size = [2*line_nums - memory_column,len_msg]))
    msg_metrics_list.append(np.random.randint(0,1,size = [line_nums,len_msg]))

    #构建第二个矩阵
    column_metrics_list = [np.random.randint(0,1,size = [int(line_nums/k_column*n_column),len_msg])]
    for i in range(int(block_nums/2-1)) :
        msg_m = msg_metrics_list[i+1]
        column_m = zeros([int((2*line_nums)/k_column*n_column),len_msg],dtype= int)
        for j in range(len_msg):
            code = cc.conv_encode(msg_m[:,j],trellis_column, termination='term')
            for k in range(code.size):
                column_m[k,j] = code[k]
        column_metrics_list.append(column_m)

    column_metrics_list.append(np.random.randint(0,1,size = [int(line_nums/k_column*n_column),len_msg]))   
    #print(len(column_metrics_list))
    #构建第三个矩阵
    codeword_metrics_list = []
    bf_nums = 0
    at_nums = 1
    for i in range(int(block_nums/2)):
        bf_metrics = column_metrics_list[bf_nums]
        at_metrics = column_metrics_list[at_nums]
        bf_nums += 1
        at_nums += 1
        #print(len_msg,memory_row,k_row,n_row)
        #time.sleep(100)
        code_metrics = zeros([int((line_nums)/k_column*n_column),int((2*len_msg+memory_row)/k_row*n_row)],dtype= int)
        #print(int((2*len_msg+memory_row)/k_row*n_row))
        for j in range(int((line_nums)/k_column*n_column)):
            if i == 0 :
                bf_msg = zeros(bf_metrics.shape[1])
            else:
                bf_msg = bf_metrics[int((line_nums)/k_column*n_column)+j]
            at_msg = at_metrics[j]
            msg_now =np.append(bf_msg,at_msg).astype(int)
            #print(msg_now)
            print(msg_now)
            #print(len(msg_now))
            code_now = cc.conv_encode(msg_now,trellis_row, termination='term')
            #if i == 1 :
            #print(msg_now )
            #print(code_now)
            #print('----------')
            for k in range(len(code_now)):
                code_metrics[j,k] = code_now[k]
        codeword_metrics_list.append(code_metrics)
    
    return [msg_metrics_list,codeword_metrics_list]



def Interleaver_2_encode(len_msg,block_nums,trellis_row,G_column,input_mode = False):

    n_row = trellis_row.n
    k_row = trellis_row.k
    memory_row = trellis_row.total_memory
    
    v_column = G_column.shape[0]
    msg_column = G_column.shape[1]

    #总共要构造3个矩阵
    #信息矩阵为v_column/2 乘 len_msg
    #纵向后的矩阵为 v_column/2 乘 len_msg
    #横向后的矩阵为 v_column/2 乘 2*len_msg/k_row*n_row 

    #构造第一个矩阵
    m_1_row = int(v_column/2)
    m_1_column = len_msg
    msg_metrics_list = [np.random.randint(0,1,size = [m_1_row,m_1_column])]
    for i in range(int(block_nums/2)-1):
        msg_metrics_list.append(np.random.randint(0,2,size = [msg_column,len_msg]))
    msg_metrics_list.append(np.random.randint(0,1,size = [m_1_row,len_msg]))
    #print(len(msg_metrics_list))

    #构建第二个矩阵
    column_metrics_list = [np.random.randint(0,1,size = [m_1_row,len_msg])]
    for i in range(int(block_nums/2-1)) :
        #print(i)
        msg_m = msg_metrics_list[i+1]
        column_m = zeros([v_column,len_msg],dtype= int)
        #print(np.shape(msg_m))
        for j in range(len_msg):
            column_m[:,j] = binaryproduct(G_column,msg_m[:,j])
        column_metrics_list.append(column_m)
    column_metrics_list.append(np.random.randint(0,1,size = [m_1_row,len_msg]))   
    #print(len(column_metrics_list))
    #构建第三个矩阵
    codeword_metrics_list = []
    bf_nums = 0
    at_nums = 1
    for i in range(int(block_nums/2)):
        bf_metrics = column_metrics_list[bf_nums]
        at_metrics = column_metrics_list[at_nums]
        bf_nums += 1
        at_nums +=1
        code_metrics = zeros([int(v_column/2),int((2*len_msg+memory_row)/k_row*n_row)],dtype= int)
        for j in range(int(v_column/2)):
            if i == 0 :
                bf_msg = zeros(bf_metrics.shape[1])
            else:
                bf_msg = bf_metrics[int(v_column/2)+j]

            at_msg = at_metrics[j]
            msg_now =np.append(bf_msg,at_msg).astype(int)
            #print(msg_now)
            code_now = cc.conv_encode(msg_now,trellis_row, termination='term')
            for k in range(len(code_now)):
                code_metrics[j,k] = code_now[k]
        codeword_metrics_list.append(code_metrics)

    return [msg_metrics_list,codeword_metrics_list]





def Interleaver_3_encode(line_nums,block_nums,G_row,trellis_column,input_mode = False):

    v_row = G_row[0]
    msg_row = G_row[1]

    n_column = int(line_nums/trellis_column.k*trellis_column.n)
    k_column = line_nums

    #

def Interleaver_3_encode(block_nums,G_row,G_column,input_mode = False):

    v_row = G_row.shape[0]
    msg_row = G_row.shape[1]

    v_column = G_column.shape[0]
    msg_column = G_column.shape[1]   

    
    msg_metrics_list = [np.random.randint(0,1,size = [int(v_row/2),int(v_column/2)])]
    #print(msg_metrics_list)
    code_metrics_list = []

    bf_metrics = np.random.randint(0,1,size = [int(v_row/2),int(v_column/2)])
    code_array_list = []
    for i in range(1,block_nums+1):
        if i %2 == 0 :
            continue
        #print(i,block_nums-1)
        if i == block_nums-1 :
            current_msg_metrics = np.random.randint(0,1,size = [int(v_row/2),msg_column-int(v_column/2)])
            #print(current_msg_metrics)
            msg_metrics_list.append(current_msg_metrics)
            
        else:
            current_msg_metrics = np.random.randint(0,2,size = [int(v_row/2),msg_column-int(v_column/2)])
            next_msg_metrics = np.random.randint(0,2,size = [msg_row-int(v_row/2),int(v_column/2)])
            #print(current_msg_metrics)
            #print(next_msg_metrics)
            msg_metrics_list.append(current_msg_metrics)
            msg_metrics_list.append(next_msg_metrics)

        #msg_metrics_list.append(current_msg_metrics)
        

        current_code_metrics = zeros([int(v_row/2),int(v_column/2)], dtype= int)
#
            #print(bf_metrics)
            #print(current_msg_metrics)
        for i in range(int(v_row/2)):
            bf_msg = bf_metrics[i]
            at_msg = current_msg_metrics[i]
            
            current_msg = np.append(bf_msg,at_msg) 
            #print(current_msg)
            current_code = binaryproduct(G_row,current_msg)
            
            current_code_metrics[i] = current_code[int(v_row/2):]
            #print(current_code)
            #print(current_code_metrics[i])

        fin_code_metrics = zeros([int(v_row/2),v_column],dtype= int)

        for i in range(int(v_row/2)):
            fin_code_metrics[i] = np.append(bf_metrics[i],current_code_metrics[i])

        

        code_array_list.append(fin_code_metrics)



        for i in range(int(v_column/2)):
            bf_msg = current_code_metrics[:,i]
            at_msg = next_msg_metrics[:,i]
            current_msg = np.append(bf_msg,at_msg)
            current_code = binaryproduct(G_column,current_msg)
            bf_metrics[:,i] = current_code[int(v_column/2):]
    #print("----------------------------------")
    #print(msg_metrics_list)
    #print("----------------------------------")
    #print(code_array_list)
        #print(code_array_list)
    return [msg_metrics_list,code_array_list]


if __name__ == '__main__':
    n = 60
    d_v = 2
    d_c = 6
    n = 60
    d_v = 2
    d_c = 4
    n = 60

    d_c = 6
    snr = 100
    H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=False,seed=1)
    print(np.shape(G))
    #Interleaver_3_encode(4,G,G,input_mode = False)
    