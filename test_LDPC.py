#本代码为LDPC的编码与解码


from platform import java_ver
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

from pyldpc import make_ldpc, encode, decode, get_message

def binaryproduct(X, Y):

    A = X.dot(Y)
    try:
        A = A.toarray()
    except AttributeError:
        pass
    return A % 2
'''
n = 6
d_v = 2
d_c = 3
snr = 100
H, G = make_ldpc(n, d_v, d_c, systematic=False, sparse=False,seed=1)
print(H)
print(G)
#print(H.shape[0],H.shape[1])
k = G.shape[1]
v = np.random.randint(2, size=k)

print(v)

print(binaryproduct(G,[1,0,1]))
print(encode(G,v,100))
get_message
y = encode(G, v, snr)
#d = decode(H, y, snr)
#x = get_message(G, d)
'''
def max_star_unit(x,y):
    output = max(x,y)+math.log(1+exp(-abs(x-y)))
    #output = max(x,y)                          ################如果要更改为max-log-MAP，那么请修改此处########################
    return output

def max_star_fuc(nums_list):
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

def _compute_gama_unit_b(b,j,d_,d,pr_dict,r):
    if d==d_ - 1 :
        return math.log(pr_dict["pd"])
    elif d >= d_ and r[d+j-1] == b :
        return math.log(1-pr_dict["ps"])+math.log(pr_dict["pt"])+(d-d_)*math.log(pr_dict["pi"]/2)
    elif d >= d_ :
        return math.log(pr_dict["ps"])+math.log(pr_dict["pt"])+(d-d_)*math.log(pr_dict["pi"]/2)

def _compute_gama_unit(j,d_,d,pr_dict,r,Le):
    if d == d_-1 :
        return math.log(pr_dict["pd"])
    if d >= d_ and r[d+j-1] == 0 :
        #print(pr_dict)
        return max_star_unit(math.log(1-pr_dict["ps"])+Le[j-1],math.log(pr_dict["ps"])-Le[j-1]/2+math.log(pr_dict["pt"])*(pr_dict["pi"]/2)**(d-d_))
    if d >=d_ and r[d+j-1] == 1 :
        return max_star_unit(math.log(pr_dict["ps"])+Le[j-1],math.log(1-pr_dict["ps"])-Le[j-1]/2+math.log(pr_dict["pt"])*(pr_dict["pi"]/2)**(d-d_))
        

def _compute_gama(len_code,codewords,Drift,d_space,d_max,d_min,gamma_b_metrics,gamma_metrics,pr_dict,L_int) :
    codeword = codewords[0]
    for t in range(1,len_code+1):
        for current_drift in range(d_min,d_max+1):
            if abs(current_drift) > t  :       #这部分是限制d的绝对值必须小于τ
                continue
            jump=2               #这部分是限制d-d_的最大值必须小于2
            for next_drift_nums in range(-1,jump+1) :
                next_drift = current_drift+next_drift_nums
                #print(t,current_drift,next_drift)
                if next_drift > d_max or next_drift < d_min :
                    continue
                #print(t,current_drift,next_drift)
                if next_drift+t > codeword.size :
                    continue
                #print(t,current_drift,next_drift)
                #print(current_drift,next_drift)
                for i in range(2) :
                    #print(i,t,current_drift,next_drift)
                    gamma_b_metrics[t,current_drift-d_min,next_drift-d_min,i] = _compute_gama_unit_b(i,t,current_drift,next_drift,pr_dict,codeword)
                    #print(gamma_b_metrics[t,current_drift-d_min,next_drift-d_min,i])
                gamma_metrics[t,current_drift-d_min,next_drift-d_min] = _compute_gama_unit(t,current_drift,next_drift,pr_dict,codeword,L_int)


def _compute_alpha(len_code,codewords,Drift,d_space,d_max,d_min,alpha_metrics,gamma_b_metrics,gamma_metrics,pr_dict,L_int) :
    codeword = codewords[0]
    for t in range(1,len_code+1):
        for current_drift in range(d_min,d_max+1):
            if abs(current_drift) > t  :       #这部分是限制d的绝对值必须小于τ
                continue
            jump=2               #这部分是限制d-d_的最大值必须小于2
            for next_drift_nums in range(-1,jump+1) :
                next_drift = current_drift+next_drift_nums
                if next_drift > d_max or next_drift < d_min :
                    continue
                if next_drift+t >= codeword.size :
                    continue
                if alpha_metrics[t-1,current_drift-d_min] == -float('inf') :
                    continue
                if alpha_metrics[t,next_drift-d_min] == -float('inf') :
                    alpha_metrics[t,next_drift-d_min] = gamma_metrics[t,current_drift-d_min,next_drift-d_min]+alpha_metrics[t-1,current_drift-d_min]
                else:
                    alpha_metrics[t,next_drift-d_min] =max_star_unit(alpha_metrics[t,next_drift-d_min],gamma_metrics[t,current_drift-d_min,next_drift-d_min]+alpha_metrics[t-1,current_drift-d_min])

def _compute_beta(len_code,codewords,Drift,d_space,d_max,d_min,beta_metrics,gamma_b_metrics,gamma_metrics,pr_dict,L_int) :
    codeword = codewords[0]
    for t in reversed(range(1,len_code)):
        for current_drift in range(d_min,d_max+1):
            if abs(current_drift) > t  :       #这部分是限制d的绝对值必须小于τ
                continue
            jump=2              #这部分是限制d-d_的最大值必须小于2
            for next_drift_nums in range(-1,jump+1) :
                next_drift = current_drift+next_drift_nums
                if next_drift > d_max or next_drift < d_min :
                    continue
                if next_drift+t >= codeword.size :
                    continue
                if beta_metrics[t+1,current_drift-d_min] == -float('inf') :
                    continue
                if beta_metrics[t,current_drift-d_min] == -float('inf') :
                    beta_metrics[t,current_drift-d_min] = gamma_metrics[t+1,current_drift-d_min,next_drift-d_min]+beta_metrics[t+1,next_drift-d_min]
                else:
                    beta_metrics[t,current_drift-d_min] =max_star_unit(beta_metrics[t,current_drift-d_min],gamma_metrics[t+1,current_drift-d_min,next_drift-d_min]+beta_metrics[t+1,next_drift-d_min]) 


def _compute_P(len_code,codewords,Drift,d_space,d_max,d_min,alpha_metrics,beta_metrics,gamma_b_metrics,gamma_metrics,L_metrics,P_array,pr_dict,L_int):
    codeword = codewords[0]
    for t in range(1,len_code+1):
        for current_drift in range(d_min,d_max+1):
            if abs(current_drift) > t  :       #这部分是限制d的绝对值必须小于τ
                continue
            jump=2              #这部分是限制d-d_的最大值必须小于2
            for next_drift_nums in range(-1,jump+1) :
                next_drift = current_drift+next_drift_nums
                if next_drift > d_max or next_drift < d_min :
                    continue
                if next_drift+t > codeword.size :
                    continue
                for i in range(2):
                    if alpha_metrics[t-1,current_drift-d_min] == -float('inf') or gamma_b_metrics[t,current_drift-d_min,next_drift-d_min,i] == -float('inf') or beta_metrics[t,next_drift-d_min] == -float('inf') or np.isnan(gamma_b_metrics[t,current_drift-d_min,next_drift-d_min,i]) == True:   
                        continue
                    #print(t,current_drift,next_drift)
                    if  L_metrics[t-1,i]  == 0 :
                        L_metrics[t-1,i] = alpha_metrics[t-1,current_drift-d_min] + gamma_b_metrics[t,current_drift-d_min,next_drift-d_min,i]+beta_metrics[t,next_drift-d_min]
                    else:
                        L_metrics[t-1,i] = max_star_unit(L_metrics[t-1,i],alpha_metrics[t-1,current_drift-d_min] + gamma_b_metrics[t,current_drift-d_min,next_drift-d_min,i]+beta_metrics[t,next_drift-d_min])
                    #print(L_metrics[t-1,i],alpha_metrics[t-1,current_drift-d_min] ,gamma_b_metrics[t,current_drift-d_min,next_drift-d_min,i].type(),beta_metrics[t,next_drift-d_min])
                    #print(L_metrics[t-1,i])
                P_array[t-1] = L_metrics[t-1,0]-L_metrics[t-1,1]

def _compute_M_N_Q(H,n_equations,n_v,M_dict,N_dict,P_array,L_int,Q_metrics):
    for i in range(n_equations):
        for j in range(n_v):
            if H[i][j] == 1 :
                M_dict[j].append(i)
                N_dict[i].append(j)
                Q_metrics[i,j]=P_array[j]+L_int[j]

def _compute_M_N_Q_ex(H,n_equations,n_v,M_dict,N_dict,P_array,L_int,Q_metrics):
    for i in range(n_equations):
        for j in range(n_v):
            if H[i][j] == 1 :
                M_dict[j].append(i)
                N_dict[i].append(j)
                Q_metrics[i,j]=L_int[j]

def _Iterative_R(H,n_equations,n_v,M_dict,N_dict,P_array,L_int,Q_metrics,R_metrics):
    for i in range(n_equations):
        for j in range(n_v):
            if H[i][j] == 1 :
                unit=1
                M_list=N_dict[i]
                for key in M_list:
                    if key == j or  Q_metrics[i,key] == 0:
                        continue
                    #print(unit,M_list,key,Q_metrics[i,key])
                    unit = unit*math.tanh(Q_metrics[i,key]/2)
                #if math.tanh(unit) == 0 :
                    #print(N_dict,Q_metrics,i,j,H)
                R_metrics[i,j]=2/math.tanh(unit)

def _Isitzero(H,LLR_tatol):
    #print(LLR_tatol)
    code_list=[]
    for i in LLR_tatol:
        if i>0 :
            code_list.append(0)
        else:
            code_list.append(1)
    #print(code_list)
    code=np.array(code_list)
    mat = np.dot(H,code)
    #print(mat)
    for i in mat :
        if i ==0 or i%2 == 0 :
            pass
        else:
            return False
    return [code,LLR_tatol]



def _compute_LLR(H,n_equations,n_v,M_dict,N_dict,P_array,L_int,Q_metrics,R_metrics,LLR_tatol,LLR_ex):
    for j in range(n_v):
        r_sum = 0
        for i in M_dict[j]:
            r_sum += R_metrics[i,j]
        LLR_tatol[j] = L_int[j]+r_sum
        LLR_ex[j] = r_sum

def _Iterative_Q(H,n_equations,n_v,M_dict,N_dict,P_array,L_int,Q_metrics,R_metrics) :
    for i in range(n_equations):
        for j in range(n_v):
            if H[i][j] == 1 :
                unit = 0
                for m in M_dict[j]:
                    if m == i :
                        continue
                    unit += R_metrics[i,j]
                Q_metrics[i,j] = P_array[j] +unit



def log_LDPC_decode(codewords,H,G,pr_dict,D=2,mode="nomal",L_int=[]):



    n=H.shape[1]      #n
    k = n-H.shape[0]  #k
    len_ = len(codewords[0])  #码的现长度
    len_code = n              #码的原长度  
    len_msg = int(len_code/n*k)      #信息位的长度
    Drift = len_ - len_code   #位移偏差
    if L_int==[]:
        L_int=zeros(len_code)    #若不设置，默认先验概率为0
    if mode != "nomal" :
        D = 0
        Drift = 0

    if Drift >=0 :
        d_max=Drift+D
        d_min=-D
    else:
        d_max=D
        d_min=Drift-D
    
    d_space=d_max-d_min+1  #飘移的空间大小

    alpha_metrics = zeros([len_code+1,d_space])
    alpha_metrics[:,:] = -float('inf')
    alpha_metrics[0,:] = -10000
    alpha_metrics[0,0-d_min] = 0

    beta_metrics = zeros([len_code+1,d_space])
    beta_metrics[:,:] = -float('inf')
    beta_metrics[len_code,:] = -10000
    beta_metrics[len_code:Drift-d_min] = 0

    gamma_b_metrics = zeros([len_code+1,d_space,d_space,2])
    gamma_b_metrics[:,:,:,:]=-10000
    gamma_metrics = zeros([len_code+1,d_space,d_space])
    gamma_metrics[:,:,:]=-10000

    L_metrics = zeros([len_code,2])
    P_array = zeros(len_code)
    if mode == "nomal" :
        _compute_gama(len_code,codewords,Drift,d_space,d_max,d_min,gamma_b_metrics,gamma_metrics,pr_dict,L_int)

        _compute_alpha(len_code,codewords,Drift,d_space,d_max,d_min,alpha_metrics,gamma_b_metrics,gamma_metrics,pr_dict,L_int)

        _compute_beta(len_code,codewords,Drift,d_space,d_max,d_min,beta_metrics,gamma_b_metrics,gamma_metrics,pr_dict,L_int)

        _compute_P(len_code,codewords,Drift,d_space,d_max,d_min,alpha_metrics,beta_metrics,gamma_b_metrics,gamma_metrics,L_metrics,P_array,pr_dict,L_int)
    else:
        P_array = L_int

    #print(P_array)
    #接下来为和积算法的流程

    n_equations = int(H.shape[0])

    n_v = int(H.shape[1])

    M_dict={}           #定义M,为校验方程的个数维
    N_dict={}           #定义N，为码字个数的维度

    Iter_nums = 0

    for i in range(n_v):
        M_dict[i] = []
    for i in range(n_equations) :
        N_dict[i] = []

    Q_metrics = zeros([n_equations,n_v])

    R_metrics = zeros([n_equations,n_v])

    LLR_tatol = zeros(n)

    now_L_int = zeros(n)

    LLR_ex = zeros(n)

    if mode == "nomal" :
        _compute_M_N_Q(H,n_equations,n_v,M_dict,N_dict,P_array,L_int,Q_metrics)
    elif mode == "exinfo" :
        _compute_M_N_Q_ex(H,n_equations,n_v,M_dict,N_dict,P_array,L_int,Q_metrics)

    #print(Q_metrics)
    _Iterative_R(H,n_equations,n_v,M_dict,N_dict,P_array,L_int,Q_metrics,R_metrics)
    #print(R_metrics)    
    _compute_LLR(H,n_equations,n_v,M_dict,N_dict,P_array,L_int,Q_metrics,R_metrics,LLR_tatol,LLR_ex)
    #print("-------##############--")
    #print(LLR_tatol,LLR_ex)
    while True :
        if _Isitzero(H,LLR_tatol) != False or Iter_nums ==10 :
            #print(LLR_tatol)
            #print(L_int)
            break
        Iter_nums +=1
        #print(Iter_nums)
        #for i in range(len_):
        #    L_int[i]=LLR_tatol[i]-L_int[i]
        if mode == "nomal" :
            for i in range(len_code):
                now_L_int[i]=LLR_tatol[i]-L_int[i]            
            _compute_gama(len_code,codewords,Drift,d_space,d_max,d_min,gamma_b_metrics,gamma_metrics,pr_dict,now_L_int)

            _compute_alpha(len_code,codewords,Drift,d_space,d_max,d_min,alpha_metrics,gamma_b_metrics,gamma_metrics,pr_dict,now_L_int)

            _compute_beta(len_code,codewords,Drift,d_space,d_max,d_min,beta_metrics,gamma_b_metrics,gamma_metrics,pr_dict,now_L_int)

            _compute_P(len_code,codewords,Drift,d_space,d_max,d_min,alpha_metrics,beta_metrics,gamma_b_metrics,gamma_metrics,L_metrics,P_array,pr_dict,now_L_int)
            
            _Iterative_Q(H,n_equations,n_v,M_dict,N_dict,P_array,LLR_tatol,Q_metrics,R_metrics)

            _Iterative_R(H,n_equations,n_v,M_dict,N_dict,P_array,LLR_tatol,Q_metrics,R_metrics)

            _compute_LLR(H,n_equations,n_v,M_dict,N_dict,P_array,LLR_tatol,Q_metrics,R_metrics,LLR_tatol,LLR_ex)
        else:
            #P_array = LLR_ex
            
            _Iterative_Q(H,n_equations,n_v,M_dict,N_dict,P_array,LLR_tatol,Q_metrics,R_metrics)

            _Iterative_R(H,n_equations,n_v,M_dict,N_dict,P_array,LLR_tatol,Q_metrics,R_metrics)

            _compute_LLR(H,n_equations,n_v,M_dict,N_dict,P_array,LLR_tatol,Q_metrics,R_metrics,LLR_tatol,LLR_ex)
            #print(LLR_tatol)
    return [LLR_tatol,LLR_ex]

def binaryproduct(X, Y):
    """Compute a matrix-matrix / vector product in Z/2Z."""
    A = X.dot(Y)
    try:
        A = A.toarray()
    except AttributeError:
        pass
    return A % 2


#########################################################################################################################
if __name__ == '__main__':
    n = 60
    d_v = 12
    d_c = 30
    snr = 100
    H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=False,seed=1)
    pr_dict ={
        "ps":0.02,
        "pd":0.0001,
        "pi":0.0001,
        "column":0}
    msg = np.array([ 26.26070573 ,-26.26070573 , 26.26070573  ,26.26070573 , 26.26070573,
    26.26070573 , 26.26070573, -26.26070573,26.26070573, -26.26070573,
    26.26070573, -26.26070573 ,-26.26070573,  26.26070573, -26.26070573,
    26.26070573 ,-26.26070573 , 26.26070573,  26.26070573, -26.26070573,
    -26.26070573 , 26.26070573 ,-26.26070573, -26.26070573, -26.26070573,
    26.26070573 ,-26.26070573 ,-26.26070573, -26.26070573, -26.26070573,
    26.26070573, -26.26070573,  26.26070573,  26.26070573,  26.26070573,
    26.26070573 , 26.26070573, -26.26070573,  26.26070573, -26.26070573,
    26.26070573 ,-26.26070573, -26.26070573,  26.26070573, -26.26070573,
    26.26070573 ,-26.26070573,  26.26070573,  26.26070573, -26.26070573,
    -26.26070573,  26.26070573, -26.26070573, -26.26070573, -26.26070573,
    26.26070573 ,-26.26070573, -26.26070573, -26.26070573, -26.26070573])


    print(log_LDPC_decode([msg],H,G,pr_dict,D= 0,mode="exinfo",L_int=msg))

