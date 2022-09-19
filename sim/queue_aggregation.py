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
import config as parameter
#from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

TILE_BIT_RATE = parameter.get_config('TILE_BIT_RATE') 
video_names = parameter.get_config('video_names') 
TILE_SVC_BIT_RATE = parameter.get_config('TILE_SVC_BIT_RATE') #叠加后的码率
alpha1 = parameter.get_config('alpha1')
alpha2 = parameter.get_config('alpha2')

QUALITY_MAX = parameter.get_config('QUALITY_MAX')
VOD_VIDEO = parameter.get_config('VOD_VIDEO')
SEGMENT_MAX = parameter.get_config('SEGMENT_MAX')
TILE_MAX = parameter.get_config('TILE_MAX')
VIDEO_NUM = parameter.get_config('VIDEO_NUM')
VOD_NUM = parameter.get_config('VOD_NUM')
SLOT = 0.5
RTT_USER = 0.0
RTT_backhaul = 0.0
RTT_fronthaul = 0.0



def plot_cluster(result):
    for cluster in result:
        remain_time = []
        average_cache_bitrate = []     
        for one in cluster:
            remain_time.append(one[0])   
            average_cache_bitrate.append(one[1])     
        plt.scatter(remain_time,average_cache_bitrate) 
    plt.title('Segment request queue clustering')
    plt.xlabel('Remain_time')
    plt.ylabel('Average_cache_bitrate')
    plt.show()  
    
def predict_segment_delay(quality,request,predict_backhaul_bandwidth,cache_table):
    viewport = request['viewport']
    video_idx = video_names.index(request['video'])
    segment_idx = request['segment_idx']
    tile_size = request['tile_size']    
    backhaul_tile_size = 0
    backhaul_dealy = 0.0
    for tile_idx in viewport:
        if cache_table[video_idx][segment_idx][tile_idx] < quality:       
            for q in range(max(cache_table[video_idx][segment_idx][tile_idx]+1,0),quality+1):
                backhaul_tile_size += tile_size[q]
            #print(cache_table[video_idx][segment_idx][tile_idx],quality,backhaul_tile_size) 
    trans_dealy = backhaul_tile_size / predict_backhaul_bandwidth
    if trans_dealy == 0:
        segment_delay = RTT_fronthaul
    else:
        segment_delay = trans_dealy + RTT_backhaul       
    return segment_delay,backhaul_tile_size


def predict_request_info(request_queue,predict_backhaul_bandwidth,cache_table):
    for idx,request in enumerate(request_queue):
        delay_info = []
        size_info = []
        for quality in range(QUALITY_MAX):
            segment_delay,backhaul_tile_size = predict_segment_delay(quality,request,predict_backhaul_bandwidth,cache_table)
            delay_info.append(segment_delay)
            size_info.append(backhaul_tile_size)
        request_queue[idx]['delay_info'] = delay_info     
        request_queue[idx]['size_info'] = size_info
        request_queue[idx]['size_ratio'] = np.sum(size_info)/(float(max(TILE_SVC_BIT_RATE))*float(TILE_MAX)/8.0)
        #print(size_info,delay_info)        
    return request_queue
'''
def request_cluster(request_queue):
    cluster = []
    for request in request_queue:
        cluster.append([request['remain_time'],np.sum(request['delay_info'])])
    km = KMeans(n_clusters=QUEUE_MAX).fit(cluster)
    center = km.cluster_centers_
    #print(km.labels_)  
    request_queue_cluster = [ [] for j in range(max(km.labels_ + 1)) ]
    result = [ [] for j in range(max(km.labels_ + 1)) ] 
    for idx,label in enumerate(km.labels_):
        request_queue_cluster[label].append(request_queue[idx])
        result[label].append(cluster[idx])
    request_queue_cluster=[x for x in request_queue_cluster if x!=[]]
    result = [x for x in result if x!=[]]          
    #plot_cluster(result)      
    request_queue_cluster.sort(key=lambda request_queue_cluster: request_queue_cluster[0]["remain_time"])   
    #print(len(request_queue_cluster),request_queue_cluster)
    for idx,one_request_queue in enumerate(request_queue_cluster):
        request_queue_cluster[idx].sort(key=operator.itemgetter('remain_time'))
    return request_queue_cluster,center
'''

def update_cluster_info(info,request):
    info['viewport'] = list(set(info['viewport']+request['viewport']))
    info['qoe_info'] = np.sum([info['qoe_info'], request['qoe_info']], axis=0).tolist()
    info['remain_time'] = min(info['remain_time'],request['remain_time'])
    info['request_num'] += 1    
    info['request_cluster'].append(request)
    return info


def init_cluster_info(request):
    info = {'video':request['video'],'segment_idx':request['segment_idx'],'viewport':request['viewport'],'qoe_info':request['qoe_info'],\
    'remain_time':request['remain_time'],'tile_size':request['tile_size'], 'request_num':1,'quality':0,'request_cluster':[request]}
    return info

def cluster_nocache(request_queue,predict_backhaul_bandwidth):
    new_request_queue = request_queue[:]        
    info_cluster = []
    for idx,request in enumerate(new_request_queue):
        request['cluster_mark'] = 2
        info_cluster.append(init_cluster_info(request))
    no_cache_table = (np.zeros((VOD_NUM,SEGMENT_MAX,TILE_MAX),dtype=int)).tolist()
    no_cache_table += (np.ones((VIDEO_NUM-VOD_NUM,SEGMENT_MAX,TILE_MAX),dtype=int)*(-1)).tolist()               
    info_cluster = predict_request_info(info_cluster,predict_backhaul_bandwidth,no_cache_table)
    info_cluster.sort(key=operator.itemgetter('remain_time'))
    return info_cluster

def cluster(request_queue,predict_backhaul_bandwidth,cache_table):
    new_request_queue = request_queue[:]        
    new_request_queue.sort(key=operator.itemgetter('remain_time'))
    vod_cluster = []
    live_cluster = []
    info_cluster = []
    for request in request_queue:
        if request['video'] in VOD_VIDEO:
            vod_cluster.append(init_cluster_info(request))
        else:
            flag = 0
            for idx,cluster in enumerate(live_cluster):
                if cluster['video'] == request['video']:
                    live_cluster[idx] = update_cluster_info(cluster,request)
                    flag = 1
                    break
            if flag == 0:
                live_cluster.append(init_cluster_info(request))
    info_cluster = vod_cluster + live_cluster
    
    info_cluster = predict_request_info(info_cluster,predict_backhaul_bandwidth,cache_table)
    for idx,cluster in enumerate(info_cluster):#更新请求
        for j,_ in enumerate(cluster['request_cluster']):
            info_cluster[idx]['request_cluster'][j]['viewport'] = cluster['viewport']
            if j == 0:
                info_cluster[idx]['request_cluster'][j]['cluster_mark'] = 0
            else:
                info_cluster[idx]['request_cluster'][j]['cluster_mark'] = 1
    info_cluster.sort(key=operator.itemgetter('remain_time'))
    return info_cluster