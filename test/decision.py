#coding:utf-8
import os
import numpy as np
import math
import shutil
import json
import operator  
import time
import uniform_allocating
import gamebased_allocating
import load_network as load_network
import multi_users as multi_users
from network import Environment
import queue_aggregation
import cache_update
import config as parameter
import pandas as pd
import random
import sys
import copy

CACHE_MAX = parameter.get_config('CACHE_MAX')
SLEEP_TIME = parameter.get_config('SLEEP_TIME')
VIDEO_NUM = parameter.get_config('VIDEO_NUM')
TILE_SVC_BIT_RATE = parameter.get_config('TILE_SVC_BIT_RATE')
TILE_MAX = parameter.get_config('TILE_MAX')
QUALITY_MAX = parameter.get_config('QUALITY_MAX')
SEGMENT_MAX = parameter.get_config('SEGMENT_MAX')


def record_response(all_response,strategy,variable,user_num,bw_capacity,bw_fluctuation,cache_max,trace_idx,zipf,i):
    if variable == 'User Number':
        folder = "response/User Number/"+str(user_num)+"/"+str(strategy)+'/'+str(i)
    elif variable == 'Average Bandwidth (Mbps)':
        folder = "response/Average Bandwidth (Mbps)/"+str(bw_capacity)+"/"+str(strategy)+'/'+str(i)
    elif variable == 'Bandwidth Fluctuation (%)':
        folder = "response/Bandwidth Fluctuation (%)/"+str(bw_fluctuation)+"/"+str(strategy)+'/'+str(i)
    elif variable == 'Cache Capacity (%)':
        folder = "response/Cache Capacity (%)/"+str(cache_max)+"/"+str(strategy)+'/'+str(i)
    elif variable == 'multi_network':
        folder = "response/multi_network/"+str(trace_idx)+"/"+str(strategy)+'/'+str(i)
    elif variable == 'zipf':
        folder = "response/zipf/"+str(zipf)+"/"+str(strategy)+'/'+str(i)   
    elif variable == 'qoe':
        folder = "response/qoe/"+str(1)+"/"+str(strategy)+'/'+str(i)    
              
    if os.path.exists(folder):
        shutil.rmtree(folder)
        os.makedirs(folder)
    else:
        os.makedirs(folder)
    for idx,user_response in enumerate(all_response):
        path = folder+"/response_"+str(idx)+".json"
        with open(path,"w") as file:
            json.dump(user_response,file)


def if_continue_playback(users):
    for user in users:
        if user.status != 'end':
            return True
    return False


def init_network(variable,bw_capacity,bw_fluctuation,trace_idx):
    file = './bw_traces/backhual.json'
    with open(file,'r') as fp:
        bw_traces = json.load(fp)
    if variable == 'Bandwidth Fluctuation (%)':
        all_cooked_bw = [bw_traces[bw_fluctuation]]
    elif variable == 'multi_network':
        all_cooked_time, all_cooked_bw, _ = load_network.load_network(cooked_network_folder = './multi_network/')
        all_cooked_bw = [all_cooked_bw[trace_idx]]
    else:
        all_cooked_time, all_cooked_bw, _ = load_network.load_network()
    #print(all_cooked_bw)
    net_env = Environment(bw_capacity = bw_capacity, all_time = np.arange(66), all_cooked_bw = all_cooked_bw)
    return net_env
    
    
def content_delivery(time_stamp,start_mark,info_cluster,users,request_queue,cache_table,user_num,request_num,net_env,all_response,average_bandwidth):
    if start_mark == 'first':
        for request in request_queue:
            if request['video_type'] == 'LIVE':
                break
        request['quality'] = 0
        info_cluster = [{'request_cluster':[request]}]  
        request, response, info_cluster = get_response(time_stamp,info_cluster)            
        segment_delay, tans_delay, backhaul_bandwidth, cache_miss, traffic_load_reduction = net_env.download_segment_time(time_stamp,request,response,cache_table)#??????segment 
        response = update_response(response,segment_delay,backhaul_bandwidth,cache_miss,traffic_load_reduction)#??????response       
        users, request_queue, response = multi_users.update_request_queue(time_stamp,users,request_queue,request,response,cache_table,user_num)
        predict_backhaul_bandwidth = backhaul_bandwidth
        average_bandwidth = backhaul_bandwidth    
        all_response[response['user_idx']].append(response)
        reward = response['qoe']
        delay = tans_delay
    else:
        reward = 0
        size = 0
        delay = 0
        while(info_cluster != []):
            #print(info_cluster[0]['quality'])        
            request, response, info_cluster = get_response(time_stamp,info_cluster)            
            segment_delay, tans_delay, backhaul_bandwidth, cache_miss, traffic_load_reduction = net_env.download_segment_time(time_stamp,request,response,cache_table)
            response = update_response(response,segment_delay,backhaul_bandwidth,cache_miss,traffic_load_reduction)#??????response       
            time_stamp = time_stamp + tans_delay    #???????????????
            users, request_queue, response = multi_users.update_request_queue(time_stamp,users,request_queue,request,response,cache_table,user_num)                    
            #predict_backhaul_bandwidth,average_bandwidth = predict_bandwidth(response,1)    
            all_response[response['user_idx']].append(response)
            reward += response['qoe']
            #print(info_cluster[0]['quality'],request['quality'],response['quality'],response['qoe'])
            size += tans_delay*backhaul_bandwidth
            delay += tans_delay
        reward = reward/request_num                                                    
        predict_backhaul_bandwidth = size/delay                    
        average_bandwidth = ((time_stamp - delay)*average_bandwidth+size)/time_stamp                                   
    return time_stamp, users, request_queue, all_response, predict_backhaul_bandwidth, average_bandwidth, reward, delay
    

def update_response(response,segment_delay,backhaul_bandwidth,cache_miss,traffic_load_reduction):
    response['segment_delay'] = segment_delay
    response['delay'] = segment_delay + response['queueing_time']
    response['backhaul_bandwidth'] = backhaul_bandwidth
    response['traffic_load_reduction'] = traffic_load_reduction 
    response['cache_miss'] = cache_miss
    #print(response['deadline']-response['delay'])
    return response

def get_response(time_stamp,info_cluster):
    request = info_cluster[0]['request_cluster'].pop(0)
    #if request['remain_time'] < 0:
        #request['quality'] = 0       
    if len(info_cluster[0]['request_cluster']) == 0:
        #info_cluster = []
        info_cluster.pop(0)                
    #print(request['video'],request['segment_idx'],request['quality'],request['remain_time'],request['size_info'],request['delay_info'],request['delay_info'],request['cluster_mark'])
    response = {'time_stamp':time_stamp,'user_idx':request['user_idx'], 'associative_rrh':request['associative_rrh'],'tile_size':request['tile_size'][request['quality']],\
    'video':request['video'], 'segment_idx':request['segment_idx'], 'viewport':request['viewport'], 'deadline':request['deadline'],\
    'queueing_time':time_stamp - request['time_stamp'], 'quality':request['quality'], 'buffer':request['buffer'], 'delay':0, 'qoe':0, 'segment_delay':0,\
    'cache_quality':request['cache_quality'],'backhaul_bandwidth':0, 'cache_miss':0, 'traffic_load_reduction':0}   
    #print(response)
    return request, response, info_cluster

def bandwidth_managing(info_cluster,predict_backhaul_bandwidth,average_bandwidth,all_size):
    max_ddl = 5.0
    min_ddl = 1.0
    min_size = min(all_size)
    max_size = max(all_size)
    ddl = info_cluster[0]['remain_time']
    presupposed_bw = min_size + (ddl - min_ddl)*(2.0*predict_backhaul_bandwidth-min_size)/(max_ddl - min_ddl)
    return presupposed_bw

def get_info(info_cluster):
    request_num = 0
    all_size = [0,0,0,0]
    for cluster in info_cluster:
        for quality in range(QUALITY_MAX):
            all_size[quality] += cluster['size_info'][quality]
        request_num += cluster['request_num']                
    return request_num, all_size

def get_qoe_info(info_cluster,predict_backhaul_bandwidth):
    qoe_info = np.ones((len(ACTION)),dtype=int)
    for idx,action in enumerate(ACTION):
        temp_cluster = copy.deepcopy(info_cluster)   
        temp_cluster = gamebased_allocating.gamebased_allocating(temp_cluster,action*predict_backhaul_bandwidth)
        for cluster in temp_cluster:
           qoe_info[idx] += cluster['qoe_info'][cluster['quality']]                             
    return qoe_info

def decision(strategy,variable,user_num,bw_capacity,bw_fluctuation,cache_max,trace_idx,zipf,i):
    start = time.time()
    #all_viewport = load_viewport.load_all_viewport() 
    #all_userview = load_userview.load_all_userview()
    time_stamp = 0
    net_env = init_network(variable,bw_capacity/1000.0,bw_fluctuation/10,trace_idx)
    if strategy == 'Offline_OPT':
        users = multi_users.init_multi_users(user_num,True)
    else:
        users = multi_users.init_multi_users(user_num,False)
    cache_table = cache_update.init_cache(cache_max*CACHE_MAX/100.0)
    request_queue = multi_users.init_request_queue(users,user_num,cache_table)
    info_cluster = []
    last_action = 0    
    average_bandwidth = 0
    request_num = 0
    all_response = [[] for j in range(user_num)]
    #????????????????????????
    time_stamp, users, request_queue, all_response, predict_backhaul_bandwidth, average_bandwidth, _, delay= content_delivery(time_stamp,'first',info_cluster,users,request_queue,cache_table,user_num,request_num,net_env,all_response,average_bandwidth) 

    while if_continue_playback(users):
    #for i in range(1000):
        if request_queue == []:#????????????????????????
            time_stamp = time_stamp + 0.1
            request_queue = multi_users.add_request(time_stamp,users,request_queue,0.1,user_num,cache_table)
        else:
            if strategy == 'Offline_OPT':
                info_cluster = queue_aggregation.cluster(request_queue,predict_backhaul_bandwidth,cache_table)
                request_num, all_size = get_info(info_cluster)                
                presupposed_bw = bandwidth_managing(info_cluster,predict_backhaul_bandwidth,average_bandwidth,all_size)
                info_cluster = gamebased_allocating.gamebased_allocating(info_cluster,presupposed_bw)              
            
            elif strategy == 'CBG_ONLY':            
                info_cluster = queue_aggregation.cluster(request_queue,predict_backhaul_bandwidth,cache_table)
                request_num, all_size = get_info(info_cluster)
                presupposed_bw = max(predict_backhaul_bandwidth*(info_cluster[0]['remain_time']*0.9),min(all_size))                
                info_cluster = gamebased_allocating.gamebased_allocating(info_cluster,presupposed_bw)  
            
            elif strategy == 'TJCD360':
                info_cluster = queue_aggregation.cluster(request_queue,predict_backhaul_bandwidth,cache_table)
                request_num, all_size = get_info(info_cluster)
                presupposed_bw = max(predict_backhaul_bandwidth*(info_cluster[0]['remain_time']*0.9),min(all_size))                                
                info_cluster = uniform_allocating.uniform_allocating(info_cluster,min(all_size),presupposed_bw)           
            
            elif strategy == 'NO_CACHE':
                info_cluster = queue_aggregation.cluster_nocache(request_queue,predict_backhaul_bandwidth)
                request_num, all_size = get_info(info_cluster)      
                presupposed_bw = bandwidth_managing(info_cluster,predict_backhaul_bandwidth,average_bandwidth,all_size)
                info_cluster = gamebased_allocating.gamebased_allocating(info_cluster,presupposed_bw)
                                                                                    
            time_stamp, users, request_queue, all_response, predict_backhaul_bandwidth, average_bandwidth, reward, delay= content_delivery(time_stamp,'other',info_cluster,users,request_queue,cache_table,user_num,request_num,net_env,all_response,average_bandwidth)               
    record_response(all_response,strategy,variable,user_num,bw_capacity,bw_fluctuation,cache_max,trace_idx,zipf,i)

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end-start)
    