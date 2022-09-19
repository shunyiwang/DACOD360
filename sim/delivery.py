#coding:utf-8
import os
import numpy as np
import math
import shutil
import json
import operator  
import load_network as load_network
import multi_users as multi_users
from network import Environment
import segment_scheduling
import bitrate_allocation
import bitrate_allocating
import cache_update
import gamebased_allocating
import config as parameter
#from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
SLOT = 1
epoch = 100
queue_config = 1
QUEUE_MAX = 5

TILE_BIT_RATE = parameter.get_config('TILE_BIT_RATE') 
QUALITY_MAX = parameter.get_config('QUALITY_MAX')


def get_info(info_cluster):
	user_num = 0
	min_bandwidth = 0
	for cluster in info_cluster:
		user_num += cluster['request_num']
		min_bandwidth += min(cluster['size_info'])
	return user_num, min_bandwidth


def content_delivery(time_stamp,users,info_cluster,cache_table,predict_backhaul_bandwidth,presupposed_bw,strategy):

    user_num, min_bandwidth = get_info(info_cluster)
    if strategy == 'proposed':
        #print(presupposed_bw)
        info_cluster = gamebased_allocating.gamebased_allocating(info_cluster,presupposed_bw)                  
        #for a in info_cluster:           
            #print(a['video'],a['request_num'],a['quality'],a['remain_time'],a['size_info'],a['delay_info'],a['qoe_info'])
        #print('$$$$$$$$$$$$$$')
        
    elif strategy == 'temporal_greedy':
        size = min_bandwidth
        for quality in range(1,QUALITY_MAX):        
            for idx,info in enumerate(info_cluster):
                if size + info['size_info'][quality] - info['size_info'][info['quality']] < presupposed_bw:
                    size += info['size_info'][quality] - info['size_info'][info['quality']]
                    info_cluster[idx]['quality'] = quality
        #for a in info_cluster:           
            #print(a['video'],a['request_num'],a['quality'],a['remain_time'],a['size_info'],a['delay_info'],a['qoe_info'])
        #print('##############')
  
    elif strategy == 'spatio_greedy':
        info_cluster = gamebased_allocating.gamebased_allocating(info_cluster,predict_backhaul_bandwidth*max(info_cluster[0]['remain_time'],0))
        print(predict_backhaul_bandwidth*max(info_cluster[0]['remain_time'],0))        
        #for a in info_cluster:
            #print(a['video'],a['request_num'],a['quality'],a['remain_time'],a['size_info'],a['delay_info'],a['qoe_info'])
        #print('.................')                        
   
    elif strategy == 'greedy':
        '''
        info_cluster = [info_cluster[0]]    
        delay_info = info_cluster[0]['delay_info']     
        #print(request['remain_time'])        
        for idx,bitrate in enumerate(TILE_BIT_RATE):
            if delay_info[idx] > info_cluster[0]['remain_time']-1:
                break
            info_cluster[0]['quality'] = idx
        #print('.................')
        #for a in info_cluster:
            #print(a['video'],a['request_num'],a['quality'],a['remain_time'],a['size_info'],a['delay_info'],a['qoe_info'])
        '''
        size = min_bandwidth
        available_bandwidth = predict_backhaul_bandwidth*max(info_cluster[0]['remain_time'],0)        
        for quality in range(1,QUALITY_MAX):
            for idx,info in enumerate(info_cluster):
                if size + info['size_info'][quality] - info['size_info'][info['quality']] < available_bandwidth:
                    size += info['size_info'][quality] - info['size_info'][info['quality']]
                    info_cluster[idx]['quality'] = quality
      	#all_size = 0
      	#for cluster in info_cluster:
      		#all_size += cluster['size_info'][cluster['quality']]
      	#print(all_size,size,predict_backhaul_bandwidth*max(info_cluster[0]['remain_time'],0))                    
                    
        print(info_cluster[0]['remain_time'],predict_backhaul_bandwidth,predict_backhaul_bandwidth*max(info_cluster[0]['remain_time'],0))       
    for idx,info in enumerate(info_cluster):#更新请求
        for j,_ in enumerate(info['request_cluster']):
            info_cluster[idx]['request_cluster'][j]['quality'] = info_cluster[idx]['quality']                             

    return info_cluster

   


