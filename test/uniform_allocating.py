#coding:utf-8
import os
import numpy as np
import math
import shutil
import json
import operator 
import config as parameter
import matplotlib.pyplot as plt

QUALITY_MAX = parameter.get_config('QUALITY_MAX')

def uniform_allocating(info_cluster,min_size,presupposed_bw):
    size = min_size
    info_cluster.sort(key=operator.itemgetter('request_num'))
    if size < presupposed_bw:    
        for quality in range(1,QUALITY_MAX):        
            for idx,info in enumerate(info_cluster):
                if size + info['size_info'][quality] - info['size_info'][info['quality']] < presupposed_bw:
                    size += info['size_info'][quality] - info['size_info'][info['quality']]
                    info_cluster[idx]['quality'] = quality
                else:
                    break
    info_cluster.sort(key=operator.itemgetter('remain_time'))
    for idx,info in enumerate(info_cluster):#更新请求
        for j,_ in enumerate(info['request_cluster']):
            info_cluster[idx]['request_cluster'][j]['quality'] = info_cluster[idx]['quality'] 
    return info_cluster

   


