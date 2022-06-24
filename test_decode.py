#本代码为全系统的解码器
from configparser import Interpolation
from http.client import OK
from test_BCJR import log_map_decode
from test_LDPC import log_LDPC_decode
from platform import java_ver
import sys

import numpy as np
from numpy import NaN, array, zeros, exp, log, empty
import math
from commpy.channelcoding import conv_encode
from commpy.utilities import dec2bitarray
import time
import commpy.channelcoding.convcode as cc
import random
random.seed(1)

import tqdm
from pyldpc import make_ldpc, encode, decode, get_message
import copy


def output_msg(input_list):
    output_list = []
    for block in input_list :
        msg_meritcs = zeros([block.shape[0],block.shape[1]],dtype = int)
        for i in range(block.shape[0]):
            for j in range(block.shape[1]):
                if block[i,j] < 0 :
                    msg_meritcs[i,j] = 1
        output_list.append(msg_meritcs)
    return output_list
                    

def Decode_conv(code_meritcs_list,pr_dict,trellis,len_msg,D,multiple_nums,line_nums):
    total_memory = trellis.total_memory
    #line_nums =int(len(code_meritcs_list)/multiple_nums)
    msg_L_meritcs = zeros([line_nums,len_msg+trellis.total_memory],dtype = int)
    msg_meritcs = zeros([line_nums,len_msg+trellis.total_memory],dtype = int)
    for code_list in code_meritcs_list :
        block_index = code_list[0]
        row_index = code_list[1]
        codewords = [code_list[2],len_msg,len_msg+total_memory]
        #print("f")
        st = time.time()
        #print(codewords)
        L_array = log_map_decode(codewords,trellis,pr_dict,D)[1]
        #print(time.time()-st)
        msg_L_meritcs[row_index,:] = msg_L_meritcs[row_index,:]+L_array
        #print("OK")

    for i in range(msg_meritcs.shape[0]):
        for j in range(msg_meritcs.shape[1]):
            if msg_L_meritcs[i,j] > 0 :
                msg_meritcs[i,j] = 0
            else :
                msg_meritcs[i,j] = 1

    return msg_meritcs

def Decode_LDPC(code_meritcs_list,pr_dict,H,G,D,multiple_nums,line_nums):
    len_code = G.shape[0]
    msg_L_meritcs = zeros([line_nums,len_code],dtype = int)
    msg_meritcs = zeros([line_nums,len_code],dtype = int)
    for code_list in code_meritcs_list :
        block_index = code_list[0]
        row_index = code_list[1]
        codewords = [code_list[2]]
        #print(codewords)
        L_array = log_LDPC_decode(codewords,H,G,pr_dict,D,mode="nomal",L_int=[])[0]
        msg_L_meritcs[row_index,:] = msg_L_meritcs[row_index,:]+L_array
    
    for i in range(msg_L_meritcs.shape[0]):
        for j in range(msg_L_meritcs.shape[1]):
            if msg_L_meritcs[i,j] > 0 :
                msg_meritcs[i,j] = 0
            else:  
                msg_meritcs[i,j] = 1 
    return msg_meritcs


######接下来交织器1和交织器2的编程思想很类似，都是先构建一个外信息矩阵，然后横行纵向疯狂在中间传递信息
def Decode_Interleaver_1(len_msg,line_nums,after_channel_code_list,pr_dict,trellis_row,trellis_column,block_nums,iterations,D):
    n_row = trellis_row.n
    k_row = trellis_row.k
    memory_row = trellis_row.total_memory

    n_column = trellis_column.n
    k_column = trellis_column.k
    memory_column = trellis_column.total_memory

    before_row_index = -1
    before_LLR = 0

    msg_block_L_list =[]
    msg_block_nums = int((block_nums/2)-1)
    for i in range(msg_block_nums):
        msg_block_L_list.append(zeros([int(2*line_nums/k_column*n_column),len_msg]))

    for nothing in range(iterations):

        ex_msg_list = copy.deepcopy(msg_block_L_list)
        msg_block_L_list =[]
        msg_block_nums = int((block_nums/2)-1)
        for i in range(msg_block_nums):
            msg_block_L_list.append(zeros([int(2*line_nums/k_column*n_column),len_msg]))
        for code_list in tqdm.tqdm(after_channel_code_list) :
            block_index = code_list[0]
            row_index = code_list[1]
            codewords = [code_list[2],2*len_msg,2*len_msg+memory_row]
            if row_index == before_row_index:
                L_int = before_LLR
            else :
                before_row_index = row_index
                if block_index == 0:
                    L_int_f =zeros([1,len_msg])
                else:
                    L_int_f = ex_msg_list[block_index-1][int(line_nums/k_column*n_column+row_index),:]
                if block_index  == msg_block_nums:
                    L_int_b = zeros([1,len_msg])
                else:
                    #print(block_index,msg_block_nums)
                    L_int_b = ex_msg_list[block_index][row_index,:]
                #print(L_int_f,L_int_b)
                L_int = np.append(L_int_f,L_int_b)
                L_int = np.append(L_int,zeros([1,memory_row]))
                #print(np.shape(L_int))
                #print(2*len_msg+memory_row)
                #print(L_int,codewords)
                #time.sleep(5)
                before_LLR = L_int

                after = log_map_decode(codewords,trellis_row,pr_dict,D,mode="nomal",L_int = L_int)
                co = after[0]
                L_array = after[1] 
            #if block_index == 1 :

            #print("----------------------")
            #print("L_int :",L_int)
                #print("code",codewords[0] )
                #print(L_array)
                #print("L",L_array[:len_msg])
                #print(co)
                #print(co[:len_msg])
                ##time.sleep(20)
            
            #####接下来要把LLR赋值到L矩阵中，然后纵向译码
            if block_index == 0 :
                msg_block_L_list[0][row_index,:] += L_array[len_msg:-memory_row]
            elif block_index  == msg_block_nums:
                #print(int(line_nums/k_column*n_column+row_index))
                #print (L_array)
                msg_block_L_list[block_index-1][int(line_nums/k_column*n_column+row_index),:] += L_array[:len_msg]
                #print(msg_block_L_list[block_index-1])
                #print(int(line_nums/k_column*n_column+row_index))
                #print( msg_block_L_list[block_index-1][int(line_nums/k_column*n_column+row_index),:])
                #time.sleep(10)
            else:    
                msg_block_L_list[block_index-1][int(line_nums/k_column*n_column+row_index),:] += L_array[:len_msg]
                msg_block_L_list[block_index][row_index,:] +=  L_array[len_msg:-memory_row]
            #print("block",msg_block_L_list[0])
        #test_=msg_block_L_list
            #time.sleep(10)      
        #print(output_msg(msg_block_L_list))
        for msg_block_index in range(len(msg_block_L_list)):
            msg_block =msg_block_L_list[msg_block_index]
            for j in range(msg_block.shape[1]):
                #print(msg_block[:,j])        
                codewords = [msg_block[:,j],int((2*line_nums - memory_column)),int(2*line_nums)]
                #print(codewords)
                if nothing == iterations - 1:
                    column_L_array = L_array = log_map_decode(codewords,trellis_column,pr_dict,D,mode="exinfo",L_int=msg_block[:,j])[1] 
                else:
                    column_L_array = L_array = log_map_decode(codewords,trellis_column,pr_dict,D,mode="exinfo",L_int=msg_block[:,j])[1] 
                #print(L_array)
                #time.sleep(10) 
                ############注意 上述函数需要重新修改一下，例如D要去掉###############
                if nothing == iterations-1 :
                    msg_block_L_list[msg_block_index][:,j] += column_L_array
                else:
                    msg_block_L_list[msg_block_index][:,j] = column_L_array 
                #print(column_L_array)
                #time.sleep(10)
        #print(output_msg(msg_block_L_list))
        fin_msg_list =[]

        
        #print(msg_block_L_list)
        #print(msg_block_L_list)
    for msg_block in msg_block_L_list:
        fin_msg_block = zeros([int(2*line_nums),len_msg])
        #print(fin_msg_block )
        t_nums = int(2*line_nums/k_column)
        for t in range(t_nums- memory_column): 
            for k in range(k_column):
                fin_msg_block[k_column*t+k,:] = msg_block[n_column*t+k,:] 
        fin_msg_list.append(fin_msg_block)
            #print(fin_msg_block)
        #print(output_msg(msg_block_L_list))
       
    return output_msg(fin_msg_list)

        
def Decode_Interleaver_2(len_msg,block_nums,after_channel_code_list,pr_dict,trellis_row,H_column,G_column,iterations,D):
    
    n_row = trellis_row.n
    k_row = trellis_row.k
    memory_row = trellis_row.total_memory
    
    v_column = G_column.shape[0]
    msg_column = G_column.shape[1]  
    msg_block_L_list = []  
    msg_block_nums = int((block_nums-1)/2)

    before_row_index = -1
    before_LLR = 0

    for i in range(msg_block_nums):
        msg_block_L_list.append(zeros([v_column,len_msg] ))

    for nothing in range(iterations):

        ex_msg_list = copy.deepcopy(msg_block_L_list)
        msg_block_L_list = []  
        msg_block_nums = int((block_nums-1)/2)
        for i in range(msg_block_nums):
            msg_block_L_list.append(zeros([v_column,len_msg] ))

        for code_list in after_channel_code_list :
            block_index = code_list[0]
            row_index = code_list[1]
            codewords = [code_list[2],2*len_msg,2*len_msg+memory_row]

            if row_index == before_row_index:
                L_int = before_LLR            
            else : 
                before_row_index = row_index
                if block_index == 0:
                    L_int_f =zeros([1,len_msg])
                else:
                    L_int_f = ex_msg_list[block_index-1][int(v_column/2+row_index),:]
                if block_index  == msg_block_nums:
                    L_int_b = zeros([1,len_msg])
                else:
                    #print(block_index,msg_block_nums)
                    L_int_b = ex_msg_list[block_index][row_index,:]
                #print(L_int_f,L_int_b)
                L_int = np.append(L_int_f,L_int_b)
                L_int = np.append(L_int,zeros([1,memory_row]))
                #print(np.shape(L_int))
                #print(2*len_msg+memory_row)
                #print(L_int,codewords)
                #time.sleep(5)
                before_LLR = L_int
            L_array = log_map_decode(codewords,trellis_row,pr_dict,D=2,mode="nomal",L_int=L_int)[1]
            #####接下来要把LLR赋值到L矩阵中，然后纵向译码
            if block_index == 0 :
                msg_block_L_list[0][row_index,:] += L_array[len_msg:-memory_row]
            elif block_index  == msg_block_nums:
                msg_block_L_list[block_index-1][int(v_column/2+row_index),:] += L_array[:len_msg]
            else:

                msg_block_L_list[block_index-1][int(v_column/2+row_index),:] += L_array[:len_msg]
                msg_block_L_list[block_index][row_index,:] += L_array[len_msg:-memory_row]
        
        #print(msg_block_L_list)
        for msg_block_index in range(len(msg_block_L_list)):
            msg_block =msg_block_L_list[msg_block_index]
            for j in range(msg_block.shape[1]):
                codewords = [msg_block[:,j]]
                #print("-----")
                #print(msg_block[:,j])
                if nothing == iterations-1 :
                    column_L_array  = log_LDPC_decode(codewords,H_column,G_column,pr_dict,D=0,mode="exinfo",L_int=msg_block[:,j])[0] 
                else: 
                    column_L_array  = log_LDPC_decode(codewords,H_column,G_column,pr_dict,D=0,mode="exinfo",L_int=msg_block[:,j])[1] 
                #print(column_L_array)
                ############注意 上述函数需要重新修改一下，例如D要去掉,增加纵向解码的信息###############
                msg_block_L_list[msg_block_index][:,j] = column_L_array 
        #print(msg_block_L_list)
    return output_msg(msg_block_L_list)
    #################注意！这里只是输出了第二矩阵的信息值，还需要译码成信息矩阵



def Decode_Interleaver_3(block_nums,after_channel_code_list,pr_dict,H_row,G_row,H_column,G_column,D,iterations):
    v_row = G_row.shape[0]
    msg_row = G_row.shape[1]

    v_column = G_column.shape[0]
    msg_column = G_column.shape[1]  


    before_row_index = -1
    before_LLR = 0
    msg_block_L_list = []  
    msg_block_nums = int((block_nums-1)/2)

    for i in range(msg_block_nums):
        msg_block_L_list.append(zeros([v_column,int(v_row/2)]  ))
    for nothing in range(iterations):

        ex_msg_list = copy.deepcopy(msg_block_L_list)
        msg_block_L_list = []  
        msg_block_nums = int((block_nums-1)/2)

        for i in range(msg_block_nums):
            msg_block_L_list.append(zeros([v_column,int(v_row/2)]  ))

        for code_list in after_channel_code_list:
            block_index = code_list[0]
            row_index = code_list[1]
            codewords = [code_list[2]]

            if row_index == before_row_index:
                L_int = before_LLR            
            else : 
                before_row_index = row_index

                if block_index == 0:
                    L_int_f =zeros([1,int(v_row/2)])
                else:
                    L_int_f = ex_msg_list[block_index-1][int(v_column/2+row_index),:]

                if block_index  == msg_block_nums:
                    L_int_b = zeros([1,int(v_row/2)])
                else:
                    #print(block_index,msg_block_nums)
                    L_int_b = ex_msg_list[block_index][row_index,:]
                #print(L_int_f,L_int_b)
                L_int = np.append(L_int_f,L_int_b)
                #print(np.shape(L_int))
                #print(2*len_msg+memory_row)
                #print(L_int,codewords)
                #time.sleep(5)
                before_LLR = L_int
            #print("------")
            #print(block_index,row_index,codewords)
            #print(codewords,L_int)      
            L_array = log_LDPC_decode(codewords,H_row,G_row,pr_dict,D=D,mode="nomal",L_int=L_int)[1] 
            #print([L_array])
            #####接下来要把LLR赋值到L矩阵中，然后纵向译码
            if block_index == 0 :
                msg_block_L_list[0][row_index,:] += L_array[int(v_row/2):]
            elif block_index  == msg_block_nums:
                msg_block_L_list[block_index-1][int(v_column/2+row_index),:] += L_array[:int(v_row/2)]
            else:

                msg_block_L_list[block_index-1][int(v_column/2+row_index),:] += L_array[:int(v_row/2)]
                msg_block_L_list[block_index][row_index,:] += L_array[int(v_row/2):]
        #print(msg_block_L_list)
        #print(output_msg(msg_block_L_list))
        for msg_block_index in range(len(msg_block_L_list)):
            msg_block =msg_block_L_list[msg_block_index]
            #print(msg_block)
            for j in range(msg_block.shape[1]):
                codewords = copy.deepcopy([msg_block[:,j]])
                #print("-----")
                #print(msg_block[:,j])
                #print(msg_block_L_list[msg_block_index][:,j])
                L_int = copy.deepcopy(msg_block[:,j])
                if nothing == iterations - 1:
                    column_L_array = log_LDPC_decode(codewords,H_column,G_column,pr_dict,D=0,mode="exinfo",L_int=L_int)[0] 
                else:
                    column_L_array = log_LDPC_decode(codewords,H_column,G_column,pr_dict,D=0,mode="exinfo",L_int=L_int)[1] 

                ############注意 上述函数需要重新修改一下，例如D要去掉,增加纵向解码的信息###############
                #print(msg_block_L_list[msg_block_index][:,j])
                msg_block_L_list[msg_block_index][:,j] = column_L_array
                #print(msg_block_L_list[msg_block_index][:,j])
                #print("----------------")
        #print(output_msg(msg_block_L_list))
    return output_msg(msg_block_L_list)
