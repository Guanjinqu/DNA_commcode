
import numpy as np


def Acc_conv(msg_metrics,decode_metrics) :
    error_nums =0
    sum_nums = 0
    for i in range(msg_metrics.shape[0]):
        for j in range(msg_metrics.shape[1]):
            sum_nums +=1 
            if msg_metrics[i,j] != decode_metrics[i,j]:
                error_nums +=1
                sum_nums +=1
    return [(sum_nums-error_nums)/sum_nums,sum_nums]

def Acc_LDPC(msg_metrics,decode_metrics) :
    print(np.shape(msg_metrics),np.shape(decode_metrics))
    error_nums =0
    sum_nums = 0
    for i in range(msg_metrics.shape[0]):
        for j in range(msg_metrics.shape[1]):
            sum_nums +=1 
            if msg_metrics[i,j] != decode_metrics[i,j]:
                error_nums +=1

    return [(sum_nums-error_nums)/sum_nums,sum_nums]  

def Acc_I1(msg_metrics_list,decode_metrics_List):
    error_nums =0
    sum_nums = 0
    for i in range(len(msg_metrics_list)):
        if i == 0 or i ==len(msg_metrics_list)-1 :
            continue
        msg_metrics = msg_metrics_list[i]
        decode_metrics = decode_metrics_List[i-1]
        for i in range(int(msg_metrics.shape[0])):
            for j in range(msg_metrics.shape[1]):
                sum_nums +=1 
                if msg_metrics[i,j] != decode_metrics[i,j]:
                    error_nums +=1

    
    return [(sum_nums-error_nums)/sum_nums,sum_nums]

def Acc_I2(msg_metrics_list,decode_metrics_List):
    error_nums =0
    sum_nums = 0
    for i in range(len(msg_metrics_list)):
        if i == 0 or i ==len(msg_metrics_list)-1 :
            continue
        msg_metrics = msg_metrics_list[i]
        decode_metrics = decode_metrics_List[i-1]
        for i in range(msg_metrics.shape[0]):
            for j in range(msg_metrics.shape[1]):
                sum_nums +=1 
                if msg_metrics[i,j] != decode_metrics[i,j]:
                    error_nums +=1

    
    return [(sum_nums-error_nums)/sum_nums,sum_nums]

def Acc_I3(msg_metrics_list,decode_metrics_List):
    error_nums =0
    sum_nums = 0    
    for index in range(len(msg_metrics_list)):
        if index == 0 or index == len(msg_metrics_list)-1 :
            continue
        if index %2 == 0 :
            decode_metrics = decode_metrics_List[int(index/2-1)]
            msg_metrics = msg_metrics_list[index]
            for i in range(msg_metrics.shape[0]):
                for j in range(msg_metrics.shape[1]):
                    sum_nums +=1 
                    if msg_metrics[i,j] != decode_metrics[int(decode_metrics.shape[0]/2+i),j]:
                        error_nums +=1
        else:
            decode_metrics = decode_metrics_List[int((index-1)/2)]
            msg_metrics = msg_metrics_list[index]            
            for i in range(msg_metrics.shape[0]):
                for j in range(msg_metrics.shape[1]):
                    sum_nums +=1 
                    if msg_metrics[i,j] != decode_metrics[int(i),j]:
                        error_nums +=1

    return [(sum_nums-error_nums)/sum_nums,sum_nums]               