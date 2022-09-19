#coding:utf-8
import os
import numpy as np
import math
import json
import operator 
import config as parameter




COOKED_CACHE_FOLDER = './cache_table/'
video_names = parameter.get_config('video_names')
SEGMENT_MAX = parameter.get_config('SEGMENT_MAX')
TILE_MAX = parameter.get_config('TILE_MAX')
RRH_MAX = parameter.get_config('RRH_MAX')
VIDEO_NUM = parameter.get_config('VIDEO_NUM')
TILE_BIT_RATE = parameter.get_config('TILE_BIT_RATE') 
TILE_SVC_BIT_RATE = parameter.get_config('TILE_SVC_BIT_RATE')
#CACHE_RRH_MAX = 200000
#CACHE_BBU_MAX = 500000
QUALITY_MAX = parameter.get_config('QUALITY_MAX')
VOD_NUM = len(parameter.get_config('VOD_VIDEO'))
CACHE_CYCLE = 5

def init_cache(CACHE_MAX):
    with open('./cache/popularity.json',"r") as file:
        tile_popularity = json.load(file)   
    with open('video_size/video_size.json','r') as fp:     
        video_size = json.load(fp)       
    cache_table = (np.zeros((VOD_NUM,SEGMENT_MAX,TILE_MAX),dtype=int)).tolist()
    cache_table += (np.ones((VIDEO_NUM-VOD_NUM,SEGMENT_MAX,TILE_MAX),dtype=int)*(-1)).tolist()
    min_size = 0
    max_size = 0    
    for segment_idx in range(SEGMENT_MAX):    
        min_size += min(video_size[segment_idx])
        max_size += sum(video_size[segment_idx])        
    cache_size = min_size*VOD_NUM*TILE_MAX
    cache_candidate = []
    for video_idx in range(VOD_NUM):
        for segment_idx in range(SEGMENT_MAX):
            for tile_idx in range(TILE_MAX):
                for quality in range(1,QUALITY_MAX):
                    popularity = tile_popularity[video_idx][segment_idx][tile_idx]
                    size = video_size[segment_idx][quality]
                    unit_value = popularity*math.log(TILE_BIT_RATE[quality]-TILE_BIT_RATE[quality-1])/size
                    cache_candidate.append({'video_idx':video_idx,'segment_idx':segment_idx,'tile_idx':tile_idx,'quality':quality,'size':size,'unit_value':unit_value})
                    #print({'video_idx':video_idx,'segment_idx':segment_idx,'tile_idx':tile_idx,'quality':quality,'size':size,'unit_value':unit_value})                    

    cache_candidate.sort(key=operator.itemgetter('unit_value'),reverse= True)
    for cache in cache_candidate:
        table_quality = cache_table[cache['video_idx']][cache['segment_idx']][cache['tile_idx']]
        cache_quality = cache['quality']
        size = 0.0        
        for quality in range(table_quality+1,cache_quality+1):
            size += video_size[cache['segment_idx']][quality]
        unit_value = cache['unit_value']
        if cache_size + size < CACHE_MAX and cache_quality > table_quality:
            cache_table[cache['video_idx']][cache['segment_idx']][cache['tile_idx']] = cache['quality']
            #print(cache,size)            
            cache_size += size
            
            
    all_size = 0
    for video_idx in range(VOD_NUM):
        for segment_idx in range(SEGMENT_MAX):
            for tile_idx in range(TILE_MAX):
                cache_quality = cache_table[video_idx][segment_idx][tile_idx]
                for quality in range(cache_quality+1):                    
                    all_size += video_size[segment_idx][quality]
    #print(min_size*VOD_NUM*TILE_MAX,max_size*VOD_NUM*TILE_MAX,all_size,cache_size,CACHE_MAX)
    #print(cache)            
    with open("cache/cache_table.json","w") as file:
        json.dump(cache_table,file)
    return cache_table






#def replace_caching(time_stamp,response,cache_table,cache_size,cache_queue,tile_idx,rrh,tile_popularity,cache_bbu_max,cache_rrh_max):




'''
def init_cache(cooked_cache_folder = COOKED_CACHE_FOLDER):
    cache_size = 0
    cache = (np.zeros((VIDEO_NUM,SEGMENT_MAX,TILE_MAX),dtype=int)).tolist()
    data = {'cache_size' : cache_size, 'cache' : cache}
    with open(str(cooked_cache_folder)+"/bbu.json","w") as file:
        json.dump(data,file)
    for rrh_idx in range(0,RRH_MAX):
        cache = (np.zeros((VIDEO_NUM,SEGMENT_MAX,TILE_MAX),dtype=int)).tolist()
        data = {'cache_size' : cache_size, 'cache' : cache}
        with open(str(cooked_cache_folder)+"/rrh_"+str(rrh_idx)+".json","w") as file:
            json.dump(data,file)
'''


def if_cache(tile_popularity):
    return True

def find_minpopularity(one_cache_queue):
    one_cache_queue.sort(key=operator.itemgetter('popularity'))
    remove_tile = one_cache_queue[0]
    return remove_tile

def find_minrequesttime(one_cache_queue):
    one_cache_queue.sort(key=operator.itemgetter('last_request_time'))
    remove_tile = one_cache_queue[0]
    return remove_tile

def update_popularity(time_stamp,tile_popularity,one_cache_queue):
    all_quality = []
    for cache in one_cache_queue:
        predict_popularity = tile_popularity[cache['video_idx']][cache['segment_idx']][cache['tile_idx']]
        #print(predict_popularity)
        cache['popularity'] = predict_popularity
        #cache['last_request_time'] = time_stamp - tile_popularity[cache['video_idx']][cache['segment_idx']][cache['tile_idx']][1]
        #print(cache['last_request_time'])
        all_quality.append(cache['quality'])
    if all_quality == []:
        mean_quality = 0
    else:
        mean_quality = np.mean(all_quality)
    return one_cache_queue,mean_quality

def replace_caching(time_stamp,response,cache_table,cache_size,cache_queue,tile_idx,rrh,tile_popularity,cache_bbu_max,cache_rrh_max):
    all_remove_tile = []
    video_idx = video_names.index(response['video'])
    segment_idx = response['segment_idx']
    quality = response['quality']
    size = response['tile_size']    
    if rrh == 0:
        cache_max = cache_bbu_max
    else:
        cache_max = cache_rrh_max
    for idx in range(RRH_MAX+1):
        if cache_table[idx][video_idx][segment_idx][tile_idx] > 0:
            for cache_tile in cache_queue[idx]:
                if cache_tile['video_idx'] == video_idx and cache_tile['segment_idx'] == segment_idx and cache_tile['tile_idx'] == tile_idx:
                    remove_tile = cache_tile
                    break
            cache_queue[idx].remove(remove_tile)
            cache_table[idx][video_idx][segment_idx][tile_idx] = 0
            cache_size[idx] -= remove_tile['size']
    while cache_size[rrh] + size > cache_max:
        remove_tile = find_minpopularity(cache_queue[rrh])
        all_remove_tile.append(remove_tile)
        cache_queue[rrh].remove(remove_tile)
        cache_table[rrh][remove_tile['video_idx']][remove_tile['segment_idx']][remove_tile['tile_idx']] = 0
        cache_size[rrh] -= remove_tile['size']
    popularity = tile_popularity[video_idx][segment_idx][tile_idx]

    #last_request_time = time_stamp - tile_popularity[video_idx][segment_idx][tile_idx][1]
    new_cache = {'video_idx':video_idx,'segment_idx':segment_idx,'tile_idx':tile_idx,'quality':quality,'popularity':popularity,'size':size}
    cache_queue[rrh].append(new_cache)
    cache_table[rrh][video_idx][segment_idx][tile_idx] = quality
    cache_size[rrh] += size
    #print(cache_size)
    return cache_table,cache_size,cache_queue,all_remove_tile,new_cache

def if_no_cached(cache_table,video_idx,segment_idx,tile_idx,quality):
    for one_cache_table in cache_table:
        if one_cache_table[video_idx][segment_idx][tile_idx] >= quality:
            return False
    return True

def cache_strategy(size,quality,popularity,cache_queue_bbu,cache_queue_rrh,mean_quality_bbu,mean_quality_rrh,cache_size_bbu,cache_size_rrh,cache_bbu_max,cache_rrh_max,strategy):
    if strategy == 'lfu':
        min_popularity_bbu = 0
        min_popularity_rrh = 0
        remove_tile_queue = []
        while cache_size_bbu + size > cache_bbu_max:
            remove_tile = find_minpopularity(cache_queue_bbu)
            min_popularity_bbu += remove_tile['popularity']*TILE_BIT_RATE[remove_tile['quality']]
            cache_size_bbu -= remove_tile['size']
            cache_queue_bbu.remove(remove_tile)
            remove_tile_queue.append(remove_tile)
        while cache_size_rrh + size > cache_rrh_max:
            remove_tile = find_minpopularity(cache_queue_rrh)
            min_popularity_rrh += remove_tile['popularity']*TILE_BIT_RATE[remove_tile['quality']]
            cache_size_rrh -= remove_tile['size']
            cache_queue_rrh.remove(remove_tile)
            remove_tile_queue.append(remove_tile)   
        if popularity > min_popularity_bbu:
            is_cache = 1
        elif popularity > min_popularity_rrh:
            is_cache = 2
        else:
            is_cache = 0
        #print(popularity,min_popularity_bbu,min_popularity_rrh,is_cache)
    return is_cache

def update_caching(time_stamp,response,cache_table,cache_size,cache_queue,tile_popularity,real_view,cache_bbu_max,cache_rrh_max,strategy):
    viewport = response['viewport']
    video_idx = video_names.index(response['video'])
    rrh = response['associative_rrh']
    segment_idx = response['segment_idx']
    quality = response['quality']
    size = response['tile_size']    
    cache_queue[0],mean_quality_bbu = update_popularity(time_stamp,tile_popularity,cache_queue[0])
    cache_queue[rrh],mean_quality_rrh = update_popularity(time_stamp,tile_popularity,cache_queue[rrh])
    for tile_idx in viewport:
        popularity = tile_popularity[video_idx][segment_idx][tile_idx]*TILE_BIT_RATE[quality]
        is_cache = 0
        if if_no_cached(cache_table,video_idx,segment_idx,tile_idx,quality):
            if cache_size[0] + size <= cache_bbu_max:
                cache_table,cache_size,cache_queue,all_remove_tile_bbu,new_cache = replace_caching(time_stamp,response,cache_table,cache_size,cache_queue,tile_idx,0,tile_popularity,cache_bbu_max,cache_rrh_max)
            elif cache_size[rrh] + size <= cache_rrh_max:
                cache_table,cache_size,cache_queue,all_remove_tile_bbu,new_cache = replace_caching(time_stamp,response,cache_table,cache_size,cache_queue,tile_idx,rrh,tile_popularity,cache_bbu_max,cache_rrh_max)
            else:        
                is_cache = cache_strategy(size,quality,popularity,cache_queue[0][:],cache_queue[rrh][:],mean_quality_bbu,mean_quality_rrh,cache_size[0],cache_size[rrh],cache_bbu_max,cache_rrh_max,strategy)
                if is_cache == 1:#bbu
                    cache_table,cache_size,cache_queue,all_remove_tile_bbu,new_cache = replace_caching(time_stamp,response,cache_table,cache_size,cache_queue,tile_idx,0,tile_popularity,cache_bbu_max,cache_rrh_max)
                    reward = predict_utility(all_remove_tile_bbu,new_cache,real_view,tile_popularity)
                elif is_cache == 2:#rrh
                    cache_table,cache_size,cache_queue,all_remove_tile_rrh,new_cache = replace_caching(time_stamp,response,cache_table,cache_size,cache_queue,tile_idx,rrh,tile_popularity,cache_bbu_max,cache_rrh_max)
                    reward = predict_utility(all_remove_tile_rrh,new_cache,real_view,tile_popularity)
    #print(cache_size)
    return cache_table,cache_size,cache_queue




def predict_utility(all_remove_tile,new_cache,real_view,tile_popularity):
    left_request_time = real_view[new_cache['video_idx']][new_cache['segment_idx']][new_cache['tile_idx']]
    new_qoe = left_request_time*math.log(TILE_SVC_BIT_RATE[new_cache['quality']])#/TILE_BIT_RATE[new_cache['quality']]
    #print(tile_popularity[new_cache['video_idx']][new_cache['segment_idx']][new_cache['tile_idx']][0],real_view[new_cache['video_idx']][new_cache['segment_idx']][new_cache['tile_idx']],left_request_time)
    old_qoe = 0
    for remove_tile in all_remove_tile:
        left_request_time = real_view[remove_tile['video_idx']][remove_tile['segment_idx']][remove_tile['tile_idx']]
        old_qoe += left_request_time*math.log(TILE_SVC_BIT_RATE[remove_tile['quality']])#/TILE_BIT_RATE[remove_tile['quality']]
    utility = new_qoe - old_qoe

    #print(all_remove_tile,new_cache,utility)
    return utility


if __name__ == '__main__':
    cache_table = init_cache(50000000)


