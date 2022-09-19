# -*-coding:utf-8 -*-
import random
import math
import matplotlib.pyplot as plt
import config as parameter
import numpy as np
import time
import operator
import config as parameter

TILE_BIT_RATE = parameter.get_config('TILE_BIT_RATE')
QUALITY_MAX = len(TILE_BIT_RATE)

def get_info(info_cluster):
	user_num = 0
	min_bandwidth = 0
	for cluster in info_cluster:
		user_num += cluster['request_num']
		min_bandwidth += min(cluster['size_info'])
	return user_num, min_bandwidth

def gamebased_allocating(info_cluster, predict_bandwidth):
    all_user_num, min_bandwidth = get_info(info_cluster)
    used_bandwidth = min_bandwidth
    bitrate_candidate = []
    if predict_bandwidth > min_bandwidth:
        for idx,cluster in enumerate(info_cluster):#博弈论
            size_ratio = cluster['size_ratio']
            allocate_bandwidth = (predict_bandwidth - min_bandwidth)*(cluster['request_num']/float(all_user_num))
            #print(cluster['request_num'],allocate_bandwidth)
            for quality,bitrate in enumerate(TILE_BIT_RATE):
                if cluster['size_info'][quality] > allocate_bandwidth:
                    break
                info_cluster[idx]['quality'] = quality
            used_bandwidth += (cluster['size_info'][info_cluster[idx]['quality']] - cluster['size_info'][0])

        for idx,cluster in enumerate(info_cluster):#贪心
            for quality in range(info_cluster[idx]['quality'] + 1,QUALITY_MAX):
                utility = math.log(cluster['qoe_info'][quality] - cluster['qoe_info'][cluster['quality']])
                size = cluster['size_info'][quality] - cluster['size_info'][cluster['quality']] + 1.0
                #print({'cluster_idx':idx,'quality':quality,'size':size,'unit_value':utility/size})      
                bitrate_candidate.append({'cluster_idx':idx,'quality':quality,'size':size,'unit_value':utility/size})
        bitrate_candidate.sort(key=operator.itemgetter('unit_value'),reverse = True)
        for candidate in bitrate_candidate:      
            cluster_idx = candidate['cluster_idx']
            quality = candidate['quality']
            size = info_cluster[cluster_idx]['size_info'][candidate['quality']] - info_cluster[cluster_idx]['size_info'][info_cluster[cluster_idx]['quality']]
            if used_bandwidth + size < predict_bandwidth and quality > info_cluster[cluster_idx]['quality']:
                used_bandwidth += size
                info_cluster[cluster_idx]['quality'] = quality


    for idx,info in enumerate(info_cluster):#更新请求
        for j,_ in enumerate(info['request_cluster']):
            info_cluster[idx]['request_cluster'][j]['quality'] = info_cluster[idx]['quality'] 
    return info_cluster

if __name__ == '__main__':
	request_queue = []
	for i in range(3):
		request = {'remain_time':4-i,'predict_segment_delay':i,'quality':0}
		request_queue.append(request)
	request_queue,utility = find_bestquality(request_queue)
	print(request_queue)