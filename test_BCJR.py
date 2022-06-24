'''

本代码为卷积码译码器,包含log-MAP和max-log-MAP算法

本代码所需环境：
numpy
scikit-commpy


'''

import sys
from re import L, X
import numpy as np
from numpy import NaN, array, zeros, exp, log, empty
import math
from commpy.channelcoding import conv_encode
from commpy.utilities import dec2bitarray
import time
import commpy.channelcoding.convcode as cc
import random
random.seed(1)


def max_star_unit(x,y):
    #计算二元MAX*函数

    output = max(x,y)+math.log(1+exp(-abs(x-y)))
    #output = max(x,y)                          ################如果要更改为max-log-MAP，那么请修改此处########################
    return output

def max_star_fuc(nums_list):
    #计算多元情况的MAX*函数

    if len(nums_list) == 1:
        return nums_list[0]
    else:
        x=nums_list[0]
        y=nums_list[1]
        nums_list=nums_list[2:]
        output=max_star_unit(x,y)
        while True:
            if nums_list ==[]:
                break
            unit=nums_list[0]
            nums_list= nums_list[1:]
            output=max_star_unit(output,unit)
        return output

def P_fuc(r_list,x_list,pr_dict,a,b,j):
    #计算P函数，为后续gamma函数值的先导计算
    
    if j == -1:
        return math.log(pr_dict["pd"])
    elif j >= 0 and r_list[a-1] == x_list[b-1]:
        return math.log(pr_dict["pt"])+j*math.log(pr_dict['pi']/2)+math.log(1-pr_dict['ps'])
    elif j >=0 :
        return math.log(pr_dict["pt"])+j*math.log(pr_dict['pi']/2)+math.log(pr_dict['ps'])

def log_F_fuc(r_list,x_list,pr_dict,a,b) :
    #计算log域下的F函数值

    #########################################注意！如果非本模型，需要修改F函数以修改gamma值的输出####################################

    if a == b ==0 :
        return 0
    elif b == 0 :
        return -10000000
    else:
        max_list=[]
        for j in range(-1,a):
            max_list.append(P_fuc(r_list,x_list,pr_dict,a,b,j)+log_F_fuc(r_list,x_list,pr_dict,a-j-1,b-1))
        return max_star_fuc(max_list)


def now_F_fuc(r_list,x_list,pr_dict,a,b):
    #计算普通域下的F函数值，本函数为测试log域下F函数的变化，暂无实际应用。

    if a == b ==0 :
        return 1
    elif b == 0 :
        return 0
    else:
        if r_list[a-1] == x_list[b-1] :
            output=0
            output+= pr_dict["pd"]*now_F_fuc(r_list,x_list,pr_dict,a,b-1)
            for j in range(a):
                output+= ((pr_dict["pi"]/4)**j)*pr_dict["pt"]*(1-pr_dict["ps"])*now_F_fuc(r_list,x_list,pr_dict,a-j-1,b-1)
        else:
            output=0
            output+= pr_dict["pd"]*now_F_fuc(r_list,x_list,pr_dict,a,b-1)
            for j in range(a):
                output+= ((pr_dict["pi"]/4)**j)*pr_dict["pt"]*(pr_dict["ps"]/3)*now_F_fuc(r_list,x_list,pr_dict,a-j-1,b-1)      
        return output   


def _compute_gamma(k,n,rate,len_msg,tao,Drift,d_space,d_max,d_min,number_states,number_inputs,gamma_metrics,codewords,L_len,trellis,D,pr_dict,L_int,mode,gamma_star_metrics):
    #计算gamma值，并导出到gamma的值空间中。同时计算gamma*的值，并导出到gamma*的值空间中

    next_state_table = trellis.next_state_table  #行表示当前状态，列表示输入，值表示输出状态
    output_table = trellis.output_table          #行表示当前状态，列表示输入，值表示输出值  

    #下面是做遍历循环，其中continue的操作是去除部分不符合条件的空间点
    for t in range(1,tao+1):
        
        for current_state in range(number_states):
            for current_input in range(number_inputs):
                next_state = next_state_table[current_state, current_input]      #当前状态 当前输入下的 下一个状态
                code_symbol = output_table[current_state, current_input]         #当前状态 当前输入下的 输出值v
                x_codeword = dec2bitarray(code_symbol, n)                    #将输出值v转化为二进制变成数组，第二位为数组的大小
                input_codeword = dec2bitarray(current_input,k)               #将输入值u转化为二进制变成数组
                for current_drift in range(d_min,d_max+1):
                    if abs(current_drift) > 2*t  :       #这部分是限制d的绝对值必须小于2τ
                        continue
                    jump=max(min(2,k),1)               #这部分是限制d-d_的最大值必须小于2
                    for next_drift_nums in range(-jump,jump+1) :
                        next_drift = current_drift+next_drift_nums
                        if next_drift > d_max or next_drift < d_min : #限制next_drift的取值
                            continue
                        r_codeword = codewords[current_drift+(t-1)*n:next_drift+t*n]
                        if r_codeword.size == 0:           
                            continue
                        if next_drift+t*n > codewords.size :
                            continue
                        for i in range(1,k+1) :
                            current_gama_star = 0
                            #print(t,current_drift,next_drift,i,n+next_drift-current_drift,n,r_codeword,x_codeword)
                            current_gama_star += log_F_fuc(r_codeword,x_codeword,pr_dict,n+next_drift-current_drift,n)
                            for j in range(1,k+1):
                                if j == i :
                                    continue
                                current_gama_star += (1-2*input_codeword[j-1])*L_int[k*(t-1)+j-1]/2
                            
                            gamma_star_metrics[t,current_state,next_state,current_drift-d_min,next_drift-d_min,i] = current_gama_star

                            current_gama = current_gama_star+(1-2*input_codeword[i-1])*L_int[k*(t-1)+i-1]/2

                            gamma_metrics[t,current_state,next_state,current_drift-d_min,next_drift-d_min] = current_gama

def _compute_gamma_ex(k,n,rate,len_msg,tao,Drift,d_space,d_max,d_min,number_states,number_inputs,gamma_metrics,codewords,L_len,trellis,D,pr_dict,L_int,mode,gamma_star_metrics):
    #计算gamma值，并导出到gamma的值空间中。同时计算gamma*的值，并导出到gamma*的值空间中

    next_state_table = trellis.next_state_table  #行表示当前状态，列表示输入，值表示输出状态
    output_table = trellis.output_table          #行表示当前状态，列表示输入，值表示输出值  

    #下面是做遍历循环，其中continue的操作是去除部分不符合条件的空间点
    for t in range(1,tao+1):
        for current_state in range(number_states):
            for current_input in range(number_inputs):
                next_state = next_state_table[current_state, current_input]      #当前状态 当前输入下的 下一个状态
                code_symbol = output_table[current_state, current_input]         #当前状态 当前输入下的 输出值v
                x_codeword = dec2bitarray(code_symbol, n)                    #将输出值v转化为二进制变成数组，第二位为数组的大小
                input_codeword = dec2bitarray(current_input,k)               #将输入值u转化为二进制变成数组
                for i in range(1,n+1) :
                    #print(t,i)
                    #print(n*(t-1)+i-1)
                    current_gama_star = 0
                    #print(t,current_drift,next_drift,i,n+next_drift-current_drift,n,r_codeword,x_codeword)
                    #current_gama_star += log_F_fuc(r_codeword,x_codeword,pr_dict,n+next_drift-current_drift,n)
                    for j in range(1,n+1):
                        if j == i :
                            continue
                        #print(j-1,n*(t-1)+j-1)
                        current_gama_star += (1-2*x_codeword[j-1])*L_int[n*(t-1)+j-1]/2
                    #print("---------")
                    #print(t,i,current_state,next_state,current_gama_star)    
                        #print((1-2*x_codeword[j-1]),L_int[n*(t-1)+j-1]/2,current_gama_star)
                    gamma_star_metrics[t,current_state,next_state,0,0,i] = current_gama_star
                    #print(t,current_state,next_state,0,0,i)
                    #print(current_gama_star)
                    current_gama = current_gama_star+(1-2*x_codeword[i-1])*L_int[n*(t-1)+i-1]/2

                    gamma_metrics[t,current_state,next_state,0,0] = current_gama
                    
    #print( gamma_metrics)


def _compute_alpha(k,n,rate,len_msg,tao,Drift,d_space,d_max,d_min,number_states,number_inputs,alpha_metrics,codewords,L_len,trellis,D,pr_dict,L_int,mode,gamma_metrics):
    #计算α的值，并导出到α的值空间中

    next_state_table = trellis.next_state_table  #行表示当前状态，列表示输入，值表示输出状态
    output_table = trellis.output_table          #行表示当前状态，列表示输入，值表示输出值 

    for t in range(1,tao+1):
        for current_state in range(number_states):
            for current_input in range(number_inputs):
                next_state = next_state_table[current_state, current_input]
                for current_drift in range(d_min,d_max+1):            
                    if abs(current_drift) > 2*t :       #这部分是限制d的绝对值必须小于2τ
                        continue
                    jump=max(min(2,k),1)               #这部分是限制d-d_的最大值必须小于2
                    for next_drift_nums in range(-jump,jump+1) :
                        next_drift = current_drift+next_drift_nums
                        if next_drift > d_max or next_drift < d_min :
                            continue
                        r_codeword = codewords[current_drift+(t-1)*n:next_drift+t*n]
                        if r_codeword.size == 0:
                            continue
                        if next_drift > d_max or next_drift < d_min :
                            continue 
                        if alpha_metrics[t-1,current_state,current_drift-d_min] == -float('inf') :
                            continue
                        if alpha_metrics[t,next_state,next_drift-d_min] == -float('inf'):
                            alpha_metrics[t,next_state,next_drift-d_min] = gamma_metrics[t,current_state,next_state,current_drift-d_min,next_drift-d_min]+alpha_metrics[t-1,current_state,current_drift-d_min]
                        else:
                            alpha_metrics[t,next_state,next_drift-d_min] = max_star_unit(alpha_metrics[t,next_state,next_drift-d_min],gamma_metrics[t,current_state,next_state,current_drift-d_min,next_drift-d_min]+alpha_metrics[t-1,current_state,current_drift-d_min])

def _compute_alpha_ex(k,n,rate,len_msg,tao,Drift,d_space,d_max,d_min,number_states,number_inputs,alpha_metrics,codewords,L_len,trellis,D,pr_dict,L_int,mode,gamma_metrics):
    #计算α的值，并导出到α的值空间中

    next_state_table = trellis.next_state_table  #行表示当前状态，列表示输入，值表示输出状态
    output_table = trellis.output_table          #行表示当前状态，列表示输入，值表示输出值 

    for t in range(1,tao+1):
        for current_state in range(number_states):
            for current_input in range(number_inputs):
                next_state = next_state_table[current_state, current_input]
                if alpha_metrics[t,next_state,0] == -float('inf'):
                    alpha_metrics[t,next_state,0] = gamma_metrics[t,current_state,next_state,0,0]+alpha_metrics[t-1,current_state,0]
                else:
                    alpha_metrics[t,next_state,0] = max_star_unit(alpha_metrics[t,next_state,0],gamma_metrics[t,current_state,next_state,0,0]+alpha_metrics[t-1,current_state,0])
                #print(alpha_metrics[t,next_state,0])
    #print(alpha_metrics)


def _compute_beta(k,n,rate,len_msg,tao,Drift,d_space,d_max,d_min,number_states,number_inputs,beta_metrics,codewords,L_len,trellis,D,pr_dict,L_int,mode,gamma_metrics):
    #计算β的值，并导出到β的值空间中
    
    next_state_table = trellis.next_state_table  #行表示当前状态，列表示输入，值表示输出状态
    output_table = trellis.output_table          #行表示当前状态，列表示输入，值表示输出值

    for t in reversed(range(1,tao)):
        for current_state in range(number_states):
            for current_input in range(number_inputs):
                next_state = next_state_table[current_state, current_input]
                for current_drift in range(d_min,d_max+1):            
                    if abs(current_drift) > 2*t :       #这部分是限制d的绝对值必须小于2τ
                        continue
                    jump=max(min(2,k),1)               #这部分是限制d-d_的最大值必须小于2
                    for next_drift_nums in range(-jump,jump+1) :
                        next_drift = current_drift+next_drift_nums
                        if next_drift > d_max or next_drift < d_min :
                            continue
                        r_codeword = codewords[current_drift+(t-1)*n:next_drift+t*n]
                        if r_codeword.size == 0:
                            continue
                        if beta_metrics[t+1,next_state,next_drift-d_min] == -float('inf'):
                            continue
                        if beta_metrics[t,current_state,current_drift-d_min] == -float('inf') :
                            beta_metrics[t,current_state,current_drift-d_min] = gamma_metrics[t+1,current_state,next_state,current_drift-d_min,next_drift-d_min]+beta_metrics[t+1,next_state,next_drift-d_min]
                        else:
                            beta_metrics[t,current_state,current_drift-d_min] = max_star_unit(beta_metrics[t,current_state,current_drift-d_min],(gamma_metrics[t+1,current_state,next_state,current_drift-d_min,next_drift-d_min]+beta_metrics[t+1,next_state,next_drift-d_min]))
def _compute_beta_ex(k,n,rate,len_msg,tao,Drift,d_space,d_max,d_min,number_states,number_inputs,beta_metrics,codewords,L_len,trellis,D,pr_dict,L_int,mode,gamma_metrics):
    #计算β的值，并导出到β的值空间中
    
    next_state_table = trellis.next_state_table  #行表示当前状态，列表示输入，值表示输出状态
    output_table = trellis.output_table          #行表示当前状态，列表示输入，值表示输出值

    for t in reversed(range(1,tao)):
        for current_state in range(number_states):
            for current_input in range(number_inputs):
                next_state = next_state_table[current_state, current_input]
                if beta_metrics[t,current_state,0] == -float('inf') :
                    beta_metrics[t,current_state,0] = gamma_metrics[t+1,current_state,next_state,0,0]+beta_metrics[t+1,next_state,0]
                else:
                    beta_metrics[t,current_state,0] = max_star_unit(beta_metrics[t,current_state,0],(gamma_metrics[t+1,current_state,next_state,0,0]+beta_metrics[t+1,next_state,0]))
                #print(beta_metrics[t,current_state,0])
def _compute_L(k,n,rate,len_msg,tao,Drift,d_space,d_max,d_min,number_states,number_inputs,codewords,L_len,trellis,D,pr_dict,L_int,mode,alpha_metrics,beta_metrics,gamma_star_metrics,L_metrics,fin_L,output_L):
    #根据α、β、γ的值，来求出LLR（log域）

    next_state_table = trellis.next_state_table  #行表示当前状态，列表示输入，值表示输出状态
    output_table = trellis.output_table          #行表示当前状态，列表示输入，值表示输出值    
    for t in range(1,tao+1):
        
        for current_state in range(number_states):
            for current_input in range(number_inputs):
                next_state = next_state_table[current_state, current_input]      #当前状态 当前输入下的 下一个状态
                code_symbol = output_table[current_state, current_input]         #当前状态 当前输入下的 输出值v
                x_codeword = dec2bitarray(code_symbol, n)                    #将输出值v转化为二进制变成数组，第二位为数组的大小

                input_codeword = dec2bitarray(current_input,k)               #将输入值u转化为二进制变成数组
                for current_drift in range(d_min,d_max+1):
                    if abs(current_drift) > 2*t :       #这部分是限制d的绝对值必须小于2τ
                        continue
                    jump=max(min(2,k),1)               #这部分是限制d-d_的最大值必须小于2
                    for next_drift_nums in range(-jump,jump+1) :
                        next_drift = current_drift+next_drift_nums   
                        if next_drift > d_max or next_drift < d_min :
                            continue
                        r_codeword = codewords[current_drift+(t-1)*n:next_drift+t*n]
                        if r_codeword.size == 0 :
                            continue   
                        for i in range(1,k+1) :
                            if (
                                alpha_metrics[t-1,current_state,current_drift-d_min]==-float('inf') or 
                                gamma_star_metrics[t,current_state,next_state,current_drift-d_min,next_drift-d_min,i] ==-float('inf') or
                                beta_metrics[t,next_state,next_drift-d_min]==-float('inf')
                            )  : 
                                
                                #print(k*(t-1)+i-1,"no")
                                continue
                            #print(k*(t-1)+i-1,"ok")
                            #print(L_metrics[k*(t-1)+i-1,0],L_metrics[k*(t-1)+i-1,1])
                            #print(input_codeword,current_state,current_input)
                            #print([t,current_state,next_state,current_drift-d_min,next_drift-d_min,i])
                            #print(input_codeword[i-1],alpha_metrics[t-1,current_state,current_drift-d_min],gamma_star_metrics[t,current_state,next_state,current_drift-d_min,next_drift-d_min,i],beta_metrics[t,next_state,next_drift-d_min])
                            if input_codeword[i-1] == 0 :
                                if L_metrics[k*(t-1)+i-1,0] == 0 :
                                    L_metrics[k*(t-1)+i-1,0]=alpha_metrics[t-1,current_state,current_drift-d_min]+gamma_star_metrics[t,current_state,next_state,current_drift-d_min,next_drift-d_min,i]+beta_metrics[t,next_state,next_drift-d_min]
                                else:
                                    L_metrics[k*(t-1)+i-1,0] = max_star_unit(L_metrics[k*(t-1)+i-1,0],
                                (alpha_metrics[t-1,current_state,current_drift-d_min]+gamma_star_metrics[t,current_state,next_state,current_drift-d_min,next_drift-d_min,i]+beta_metrics[t,next_state,next_drift-d_min]))
                                
                                    
                            if input_codeword[i-1] == 1 :
                                if L_metrics[k*(t-1)+i-1,1] == 0 :
                                    L_metrics[k*(t-1)+i-1,1] = alpha_metrics[t-1,current_state,current_drift-d_min]+gamma_star_metrics[t,current_state,next_state,current_drift-d_min,next_drift-d_min,i]+beta_metrics[t,next_state,next_drift-d_min]
                                else:
                                    L_metrics[k*(t-1)+i-1,1] = max_star_unit(L_metrics[k*(t-1)+i-1,1],
                                (alpha_metrics[t-1,current_state,current_drift-d_min]+gamma_star_metrics[t,current_state,next_state,current_drift-d_min,next_drift-d_min,i]+beta_metrics[t,next_state,next_drift-d_min]))
                            #print(L_metrics[k*(t-1)+i-1,0],L_metrics[k*(t-1)+i-1,1])
            fin_L[k*(t-1)+i-1] = L_metrics[k*(t-1)+i-1,0] - L_metrics[k*(t-1)+i-1,1]  
            output_L[k*(t-1)+i-1] = L_metrics[k*(t-1)+i-1,0] - L_metrics[k*(t-1)+i-1,1] + L_int[k*(t-1)+i-1]
            #print(k*(t-1)+i-1)





def _compute_L_ex(k,n,rate,len_msg,tao,Drift,d_space,d_max,d_min,number_states,number_inputs,codewords,L_len,trellis,D,pr_dict,L_int,mode,alpha_metrics,beta_metrics,gamma_star_metrics,L_metrics,fin_L,output_L):
    #根据α、β、γ的值，来求出LLR（log域）

    next_state_table = trellis.next_state_table  #行表示当前状态，列表示输入，值表示输出状态
    output_table = trellis.output_table          #行表示当前状态，列表示输入，值表示输出值    
    for t in range(1,tao+1):
        
            #print(i)
        for current_state in range(number_states):
            for current_input in range(number_inputs):
                
                next_state = next_state_table[current_state, current_input]      #当前状态 当前输入下的 下一个状态
                code_symbol = output_table[current_state, current_input]         #当前状态 当前输入下的 输出值v
                x_codeword = dec2bitarray(code_symbol, n)                    #将输出值v转化为二进制变成数组，第二位为数组的大小

                input_codeword = dec2bitarray(current_input,k)               #将输入值u转化为二进制变成数组
                for i in range(1,n+1) :
                    #print("----------")
                    #print(t,i,n*(t-1)+i-1,current_state,current_input,next_state,x_codeword[i-1]) 
                    #print(alpha_metrics[t-1,current_state,0],gamma_star_metrics[t,current_state,next_state,0,0,i],beta_metrics[t,next_state,0])
                    if x_codeword[i-1] == 0 :
                        if L_metrics[n*(t-1)+i-1,0] == 0 :
                            L_metrics[n*(t-1)+i-1,0]=alpha_metrics[t-1,current_state,0]+gamma_star_metrics[t,current_state,next_state,0,0,i]+beta_metrics[t,next_state,0]
                        else:
                            L_metrics[n*(t-1)+i-1,0] = max_star_unit(L_metrics[n*(t-1)+i-1,0],
                        (alpha_metrics[t-1,current_state,0]+gamma_star_metrics[t,current_state,next_state,0,0,i]+beta_metrics[t,next_state,0]))
                            
                    if x_codeword[i-1] == 1 :
                        if L_metrics[n*(t-1)+i-1,1] == 0 :
                            L_metrics[n*(t-1)+i-1,1] = alpha_metrics[t-1,current_state,0]+gamma_star_metrics[t,current_state,next_state,0,0,i]+beta_metrics[t,next_state,0]
                        else:
                            L_metrics[n*(t-1)+i-1,1] = max_star_unit(L_metrics[n*(t-1)+i-1,1],
                        (alpha_metrics[t-1,current_state,0]+gamma_star_metrics[t,current_state,next_state,0,0,i]+beta_metrics[t,next_state,0]))
                    #print(L_metrics[n*(t-1)+i-1,0],L_metrics[n*(t-1)+i-1,1])  

    for t in range(1,tao+1):
        for i in range(1,n+1) :
            fin_L[n*(t-1)+i-1] = L_metrics[n*(t-1)+i-1,0] - L_metrics[n*(t-1)+i-1,1]  
            output_L[n*(t-1)+i-1] = fin_L[n*(t-1)+i-1] +L_int[n*(t-1)+i-1]

             
def log_map_decode(codewords_list,trellis,pr_dict,D=2,mode="nomal",L_int=[]):
    """
    卷积码编码器

    负责将码字通过log——MAP译码算法译码成原始信息,同时允许输出L值以进行迭代译码。

    --------------------------------------------------------------------------
    输入：

    codewords_list: list [codewords,len_msg.len_U]      码字列表
                    codewords:  经过信道后的码字
                    len_msg :   原始信息位的长度
                    len_U :     结尾清零的信息位长度

    trellis :  class                                    卷积码编码器

    pr_dict : list  {"pt","pr","pi","pd"}               错误率字典
            
    D : int                                             飘移值的冗余阈值

    mode : "nomal","exinfo"                                         暂时没用

    L_int : list                                        先验概率
    ---------------------------------------------------------------------------


    ---------------------------------------------------------------------------
    参数：

    太多了懒得写了，重要参数我一般后边都标了，再不懂就问我吧:D
    ---------------------------------------------------------------------------

    ---------------------------------------------------------------------------
    输出：

    [msg,L_list]                二元列表，其中
                msg :           一维数组，译码后的信息位
                L_list :        二维数组， 译码后的LLR值数组
    ---------------------------------------------------------------------------


    """
    
    
    
    codewords=codewords_list[0]
    
    
    k = trellis.k  #输入bit数
    n = trellis.n  #输出bit数
    rate = float(k)/n   #码率
    L_len = int(codewords_list[2]/rate)    #v的长度
    len_msg=int(codewords_list[2])         #u的长度
    len_msg_true= int(codewords_list[1])
    tao = int(L_len/n)    #τ的长度
    Drift = len(codewords) - L_len #位移差值
    if  mode== "exinfo" :
        Drift = 0
        D = 0
    if  L_int==[]:
        L_int=zeros(len_msg)
    #print(codewords_list,L_int,tao,len_msg,L_len)
    if Drift >=0 :
        d_max=Drift+D
        d_min=-D
    else:
        d_max=D
        d_min=Drift-D
    
    d_space=d_max-d_min+1  #飘移的空间大小
    #print(d_max,d_min)
    number_states = trellis.number_states #状态个数
    number_inputs = trellis.number_inputs #trellis中每个状态的分支数量。

    alpha_metrics = zeros([tao+1,number_states,d_space])    #定义alpha的值空间，信息位x状态位xd空间
    alpha_metrics[:,:,:] = -float('inf')
    alpha_metrics[0,:,:] = -10000000
    alpha_metrics[0,0,00-d_min] = 0
    

    beta_metrics = zeros([tao+1,number_states,d_space])    #定义beta的值空间，信息位x状态位xd空间
    beta_metrics[:,:,:] = -float('inf')
    beta_metrics[tao,:,:] = -10000000
    beta_metrics[tao,0,Drift-d_min] = 0


    gamma_metrics = zeros([tao+1,number_states,number_states,d_space,d_space]) #定义gama的值空间，信息位x状态位xd空间xk
    gamma_star_metrics = zeros([tao+1,number_states,number_states,d_space,d_space,k+1]) #定义gama*的值空间，信息位x状态位xd空间xk
    if mode == "exinfo" :
        gamma_star_metrics = zeros([tao+1,number_states,number_states,d_space,d_space,n+1]) #定义gama*的值空间，信息位x状态位xd空间xk        
    gamma_metrics[:,:,:,:,:]=-10000000
    gamma_star_metrics[:,:,:,:,:,:]=-10000000
    

    if mode == "nomal" :
        L_metrics = zeros([len_msg,2])
        fin_L=zeros(len_msg)
        output_L = zeros(len_msg)
    else:
        L_metrics = zeros([L_len,2])   
        fin_L=zeros(L_len)
        output_L = zeros(L_len)
    #L_metrics[:,:]=-10000000


    if mode == "nomal" :
        _compute_gamma(k,n,rate,len_msg,tao,Drift,d_space,d_max,d_min,
                    number_states,number_inputs,gamma_metrics,codewords,
                    L_len,trellis,D,pr_dict,L_int,mode,gamma_star_metrics)
        _compute_alpha(k,n,rate,len_msg,tao,Drift,d_space,d_max,d_min,
                  number_states,number_inputs,alpha_metrics,codewords,
                  L_len,trellis,D,pr_dict,L_int,mode,gamma_metrics)

        _compute_beta(k,n,rate,len_msg,tao,Drift,d_space,d_max,d_min,
                  number_states,number_inputs,beta_metrics,codewords,
                  L_len,trellis,D,pr_dict,L_int,mode,gamma_metrics)
        _compute_L(k,n,rate,len_msg,tao,Drift,d_space,d_max,d_min,
                    number_states,number_inputs,codewords,L_len,trellis,
                    D,pr_dict,L_int,mode,alpha_metrics,beta_metrics,
                    gamma_star_metrics,L_metrics,fin_L,output_L)
    elif mode == "exinfo" :
        _compute_gamma_ex(k,n,rate,len_msg,tao,Drift,d_space,d_max,d_min,
                    number_states,number_inputs,gamma_metrics,codewords,
                    L_len,trellis,D,pr_dict,L_int,mode,gamma_star_metrics)        
        _compute_alpha_ex(k,n,rate,len_msg,tao,Drift,d_space,d_max,d_min,
                  number_states,number_inputs,alpha_metrics,codewords,
                  L_len,trellis,D,pr_dict,L_int,mode,gamma_metrics)

        _compute_beta_ex(k,n,rate,len_msg,tao,Drift,d_space,d_max,d_min,
                  number_states,number_inputs,beta_metrics,codewords,
                  L_len,trellis,D,pr_dict,L_int,mode,gamma_metrics)  
        _compute_L_ex(k,n,rate,len_msg,tao,Drift,d_space,d_max,d_min,
                    number_states,number_inputs,codewords,L_len,trellis,
                    D,pr_dict,L_int,mode,alpha_metrics,beta_metrics,
                    gamma_star_metrics,L_metrics,fin_L,output_L)   


    fin_list=[]
    for i in range(len_msg_true):
        if output_L[i] > 0 :
            fin_list.append(0)
        else:
            fin_list.append(1)
    #print(gamma_star_metrics)
    #print(alpha_metrics)
    #print( beta_metrics)
    return [np.array(fin_list),fin_L,output_L]
    #print(alpha_metrics)
    #print( beta_metrics)
    

###################################以下函数为检验译码器时随手写的几个函数，正式代码中将不存在于本代码中######################################


def get_encode(msg,trellis):
    #一个套壳的编码器

    len_msg=len(msg)
    code=cc.conv_encode(msg,trellis, termination='term')
    len_L = code.size/trellis.n*trellis.k
    return([code,int(len_msg),int(len_L)])


def channel_model(list,pr_dict):
    #信道模拟函数

    unit_list=[1,0]
    pr_dict["pt"]=((1-pr_dict["pi"])*(1-pr_dict["pd"]))
    new_pr_dict={}
    for key in pr_dict:
        new_pr_dict[key]=10000*pr_dict[key]

    code = list[0]
    af_code=[]

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
    #for i in range(len(af_code)):
    #    af_code_array[i] = str(af_code[i])

    return([af_code_array,list[1],list[2]])

def hamming(x,y):
    #输出基于汉明距离的准确率

    len_=x.size
    num_=0
    for i in range(len_):
        if x[i]== y[i]:
            num_+=1
    print((num_/len_))



if __name__ == '__main__':

    #########################################
    #以下为构建卷积码编码器
    memory = array([1])                     #寄存器的个数

    g_matrix = array([[1,2]])               #生成矩阵的样子

    fd = array([[3]])

    trellis = cc.Trellis(memory, g_matrix,feedback=fd,code_type='rsc')  #构造编码器

    memory = array([2])
    g_matrix = array([[5,7,3],[5,7,3]])
    #trellis = cc.Trellis(memory, g_matrix)
    #trellis.visualize()                    #这个函数可以可视化状态转移图

    #print(trellis.next_state_table)        #行表示当前状态，列表示输入，值表示输出状态

    #print(trellis.output_table)            #行表示当前状态，列表示输入，值表示输出值 
    #########################################


    #下为构建错误率字典
    pr_dict={"ps":0.02,"pd":0.0001,"pi":0.0001} 
    pr_dict["pt"]=((1-pr_dict["pi"])*(1-pr_dict["pd"]))

    #下为原始信息列
    msg = array([1,0,0,1,0,0,0,1,0,1,1,1,0,1,0,1,0,0,1,0,0,1,0,0,0,1,0,1,1,1,0,1,0,1,0,0,1,0,0,0,1,0,1,1,0,0,1,0,0,0,1,0,1,1,1,0,1,0,1,0,0,1,0,0,1,0,0,0,1,0,1,1,1,0,1,0,1,0,0,1,0,0,0,1,0,1,1,1,0,1,0,1,0,0,1,0,0,0,0,1,0,0,0,1,0,1,1,1,0,1,0,1,0,0,1,0,0,1,0,0,0,1,0,1,1,1,0,1,0,1,0,0,1,0,0,0,1,0,1,1,1,0,1,0,1,0,0,1,0,0,1,0,1,0,1,0,0,1,0,0])
    msg2 = array([0., 0., 0., 0., 0., 0. ,0. ,0. ,0., 0. ,1., 0., 1., 0., 1. ,0. ,1., 0., 1. ,0.],dtype= int)


    #print(len(msg))
    #下为生成码字
    #print(cc.conv_encode(msg2,trellis, termination='term'))
    code_list = get_encode(msg2,trellis) 
    #return：[数组格式的编码v，信息位的长度，总体信息位的长度]

    #下为通过信道
    after_channel_code =  channel_model(code_list,pr_dict)
    print(msg2)
    print(after_channel_code)
    #return：[通过信道后数组格式的码字r，信息位的长度，总体信息位的长度]
    

    print(log_map_decode(after_channel_code,trellis,pr_dict,D=0))

    #下为输出码字通过信道后的正确率
    hamming(code_list[0],after_channel_code[0])
    #print(after_channel_code)
    hamming(msg2,log_map_decode(after_channel_code,trellis,pr_dict,D=0)[0])
    #print(trellis.next_state_table)






