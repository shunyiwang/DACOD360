#coding:utf-8
import os
import json
import codecs
import csv
import math
import numpy as np
import pandas as pd
import config as parameter
import random
import matplotlib.pyplot as plt
import shutil

TILE_SVC_BIT_RATE = parameter.get_config('TILE_SVC_BIT_RATE')

def load_response(folder):
    cooked_files = os.listdir(folder)
    all_qoe = []
    all_quality = []
    all_cache_quality = []    
    all_miss_deadline = []
    all_cahce_hit = []
    users_segment_quality = []
    all_remain_time = []
    all_traffic = []
    for cooked_file in cooked_files:
        segment_quality = []
        cache_quality = []
        traffic = []               
        segment_qoe = []
        segment_miss = []
        remain_time = []
        backhaul_bandwidth = []
        time_stamp = []
        segment_delay = []
        segment_buffer = []              
        video_name = []                        
        tile_num = 0
        cache_miss = 0
        segment_num = 0
        cache_hit = []
        # print file_path
        file_path = folder + cooked_file       
        with open(file_path,"r") as file:
            response_list = json.load(file)
            for response in response_list:
                user_idx = response['user_idx']                                        
                segment_quality.append(TILE_SVC_BIT_RATE[response['quality']])
                segment_delay.append(response['segment_delay'])
                video_name.append(response['video'])
                traffic.append(response['traffic_load_reduction']/1000000.0)                
                backhaul_bandwidth.append(response['backhaul_bandwidth'])            
                cache_quality.append(response['cache_quality'])
                cache_hit.append((len(response['viewport']) - response['cache_miss'])/float(len(response['viewport'])))           
                segment_qoe.append(response['qoe'])
                time_stamp.append(response['time_stamp'])
                segment_buffer.append(response['buffer'])               
                segment_num += 1
                if response['deadline'] < response['delay']:
                    segment_miss.append(1.0)
                else:
                    segment_miss.append(0.0)
                remain_time.append(max(response['deadline']-response['delay'],0))          
        data = {'time_stamp':time_stamp,'qoe':segment_qoe,'quality':segment_quality,'miss_ddl':segment_miss,'cache_hit':cache_hit,'remain_time':remain_time,'cache_quality':cache_quality,'backhaul_bandwidth':backhaul_bandwidth,'video_name':video_name,'delay':segment_delay,'buffer':segment_buffer,'traffic_load_reduction':traffic}
        csv_path = './csv' + folder      
        if os.path.exists(csv_path) != True:
            os.makedirs(csv_path)               
        write_csv(data, csv_path + str(user_idx) + '.csv')                
        all_qoe.append(np.mean(segment_qoe))
        users_segment_quality.append(segment_quality)
        all_cache_quality.append(np.mean(cache_quality))
        all_quality.append(np.mean(segment_quality))
        all_miss_deadline.append(float(sum(segment_miss)/segment_num))
        all_cahce_hit.append(np.mean(cache_hit))
        all_remain_time.append(float(np.mean(remain_time)))
        all_traffic.append(sum(traffic))
    #plot_quality(users_segment_quality,5)
    return all_qoe, all_quality, all_miss_deadline, all_cahce_hit, all_remain_time, all_cache_quality, all_traffic

def write_csv(data,file):
    df = pd.DataFrame(data)
    df.to_csv(file)

def continued_product(all_qoe):
    total = 0
    jain = 0
    for qoe in all_qoe:
        total += qoe
        jain += qoe*qoe
    total = (total/len(all_qoe))
    jain = (sum(all_qoe)*sum(all_qoe)) / (jain * len(all_qoe))
    return np.mean(all_qoe),jain

def plot_quality(users_segment_quality,num):
    random.shuffle(users_segment_quality)
    for i in range(num):
        print(users_segment_quality[i])
        plt.plot(users_segment_quality[i])
    plt.xlabel('segment_idx')  
    plt.ylabel('quality')  
    plt.title('users_segment_quality')
    plt.show()



def result_analyse(strategy,variable):
    folder = "./response/" + variable
    files = os.listdir(folder)
    multi_qoe = []
    multi_jain = []
    multi_quality = []
    multi_miss = []
    multi_hit = []
    multi_remain = []
    multi_traffic = []    
    var = []
    for file in files:
        path = folder + '/' + file + '/' + str(strategy) + '/'
        var.append(float(file))
        rounds = os.listdir(path)
        all_qoe = []
        all_jain = []
        all_quality = []
        all_miss_deadline = []
        all_cahce_hit = []
        all_remain_time = []
        all_traffic = []
        for i in rounds:
            user_qoe, quality, miss_deadline, cache_hit, remain_time, cache_quality, traffic = load_response(path+i+'/')
            qoe,jain = continued_product(user_qoe)
            all_qoe.append(qoe)
            all_jain.append(jain)
            all_quality.append(np.mean(quality))
            all_miss_deadline.append(np.mean(miss_deadline))
            all_cahce_hit.append(np.mean(cache_hit))
            all_remain_time.append(np.mean(remain_time))
            all_traffic.append(sum(traffic))            
            result_folder = './result/'+ variable + '/' + file + '/' + i
            if os.path.exists(result_folder) != True:
		            os.makedirs(result_folder) 
            result_file = result_folder + '/' + str(strategy)  + '.csv'
            data = {'qoe':user_qoe,'quality':quality,'miss':miss_deadline,'cache_hit':cache_hit,'remain_time':remain_time,'cache_quality':cache_quality,'all_traffic':traffic}           
            write_csv(data, result_file)
        #print(all_qoe)
        multi_qoe.append(np.mean(all_qoe))
        multi_jain.append(np.mean(jain))
        multi_quality.append(np.mean(all_quality))
        multi_miss.append(np.mean(all_miss_deadline))
        multi_hit.append(np.mean(all_cahce_hit))
        multi_remain.append(np.mean(all_remain_time))
        multi_traffic.append(np.mean(all_traffic))
    zip_pop = zip(var,multi_qoe,multi_jain,multi_quality,multi_miss,multi_hit,multi_remain,multi_traffic)
    sort_zipped = sorted(zip_pop,key=lambda x:(x[0]))
    result = zip(*sort_zipped)
    var,multi_qoe,multi_jain,multi_quality,multi_miss,multi_hit,multi_remain,multi_traffic = [list(x) for x in result]
    return var,multi_qoe,multi_jain,multi_quality,multi_miss,multi_hit,multi_remain,multi_traffic
        


if __name__ == '__main__':
    all_qoe, all_quality, all_miss_deadline, all_cahce_hit = load_response('./response/response_proposed/')
    write_csv(all_qoe, all_quality, all_miss_deadline,all_cahce_hit,'./result/result_proposed/result.csv')
    print(continued_product(all_qoe,10),np.mean(all_quality),np.mean(all_miss_deadline),np.mean(all_cahce_hit))

