#coding:utf-8
import os
import numpy as np
import math
import shutil
import json
import operator  
import time
import gamebased_allocating
import load_viewport
import load_userview
import load_network as load_network
import multi_users as multi_users
from network import Environment
import queue_aggregation
import cache_update
import config as parameter
import logging
import multiprocessing as mp     #multiprocessing包是Python中的多进程管理包
os.environ['CUDA_VISIBLE_DEVICES']=''
import tensorflow as tf
import a3c
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


S_INFO = 5  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 5 #action
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
NUM_AGENTS = 16 #agent数目
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100   #每100次保存一次RL模型
RANDOM_SEED = 42
RAND_RANGE = 1000
TEST_LOG_FOLDER = './test_results/'
LOG_FILE = './test_results/log_sim_rl'
TRAIN_TRACES = './cooked_traces/'
#BW_CAPACITY = [100,200,300,400,500,600,700]
#BW_CAPACITY = [100,200,300,400,500,600,700]
ACTION = [0.0,0.5,1.0,1.5,2.0]
NN_MODEL = './model/nn_model_ep_6900.ckpt'
#NN_MODEL = None


#NN_MODEL = sys.argv[1]   #sys.argv[]从程序外部获取模型



def record_response(all_response,strategy,variable,user_num,bw_capacity,bw_fluctuation,cache_max,zipf,i):
    if variable == 'User num':
        folder = "response/User num/"+str(user_num)+"/"+str(strategy)+'/'+str(i)
    elif variable == 'Bandwidth capacity':
        folder = "response/Bandwidth capacity/"+str(bw_capacity)+"/"+str(strategy)+'/'+str(i)
    elif variable == 'Bandwidth fluctuation':
        folder = "response/Bandwidth fluctuation/"+str(bw_fluctuation)+"/"+str(strategy)+'/'+str(i)
    elif variable == 'Cache capacity':
        folder = "response/Cache capacity/"+str(cache_max)+"/"+str(strategy)+'/'+str(i)
    elif variable == 'Zipf parameter':
        folder = "response/Zipf parameter/"+str(zipf)+"/"+str(strategy)+'/'+str(i)
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


def init_network(network_select,bw_capacity,bw_fluctuation):
    all_cooked_time, all_cooked_bw, _ = load_network.load_network()
    file = './bw_traces/backhual.json'
    with open(file,'r') as fp:
        bw_traces = json.load(fp)
    if network_select == 'simulated':
        all_cooked_bw = [bw_traces[bw_fluctuation]]
    #print(all_cooked_bw)
    net_env = Environment(bw_capacity = bw_capacity, all_time = all_cooked_time, all_cooked_bw = all_cooked_bw)
    return net_env
    
    
def content_delivery(time_stamp,start_mark,info_cluster,users,request_queue,cache_table,user_num,request_num,net_env,all_response,average_bandwidth):
    if start_mark == 'first':
        for request in request_queue:
            if request['video_type'] == 'LIVE':
                break
        request['quality'] = 0
        info_cluster = [{'request_cluster':[request]}]  
        request, response, info_cluster = get_response(time_stamp,info_cluster)            
        segment_delay, tans_delay, backhaul_bandwidth, cache_miss, traffic_load_reduction = net_env.download_segment_time(time_stamp,request,response,cache_table)#下载segment 
        response = update_response(response,segment_delay,backhaul_bandwidth,cache_miss,traffic_load_reduction)#更新response       
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
            response = update_response(response,segment_delay,backhaul_bandwidth,cache_miss,traffic_load_reduction)#更新response       
            time_stamp = time_stamp + tans_delay    #更新时间戳
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

def rl_decision(strategy,variable,user_num,bw_capacity,bw_fluctuation,cache_max,zipf,i):            
    np.random.seed(RANDOM_SEED)
    tf.reset_default_graph()
    with tf.Session() as sess:

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)

        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters

        #恢复模型参数，为后续测试做准备
        # restore neural net parameters
        if NN_MODEL is not None:  # NN_MODEL is the path to file
            saver.restore(sess, NN_MODEL)
            print("Testing model restored.")

        ##time_stamp = 0

        s_batch = []
        a_batch = []
        r_batch = []  #仅用于记录reward
        entropy_record = []


        #反复测试，直至测试完所有视频
        #all_viewport = load_viewport.load_all_viewport() 
        #all_userview = load_userview.load_all_userview()
        time_stamp = 0
        net_env = init_network('real',bw_capacity/1000.0,bw_fluctuation/10)
        users = multi_users.init_multi_users(user_num)
        cache_table = cache_update.init_cache(cache_max*CACHE_MAX/100.0)
        request_queue = multi_users.init_request_queue(users,user_num,cache_table)
        info_cluster = []
        last_action = 0    
        average_bandwidth = 0
        request_num = 0
        all_response = [[] for j in range(user_num)]
        #初始化第一个请求
        time_stamp, users, request_queue, all_response, predict_backhaul_bandwidth, average_bandwidth, _, delay= content_delivery(time_stamp,'first',info_cluster,users,request_queue,cache_table,user_num,request_num,net_env,all_response,average_bandwidth) 

        #载入视频流训练agent,直至视频结束  
        while True:  # experience video streaming forever
            if len(s_batch) == 0:
                state = np.zeros((S_INFO, S_LEN))
            else:
                state = np.array(s_batch[-1], copy=True)
            # dequeue history record
            state = np.roll(state, -1, axis=1)
            
            info_cluster = queue_aggregation.cluster(request_queue,predict_backhaul_bandwidth,cache_table)
            request_num, all_size = get_info(info_cluster)
            temp_cluster = copy.deepcopy(info_cluster)         
            qoe_info = get_qoe_info(temp_cluster,predict_backhaul_bandwidth)             
            min_remain_time = info_cluster[0]['remain_time']
            segment_idx = info_cluster[0]['segment_idx']
            ALL_SIZE = (float(user_num*max(TILE_SVC_BIT_RATE))*float(TILE_MAX)/8.0)
            predict_delay = ACTION[last_action] 



            # this should be S_INFO number of terms
            state[0, -1] = predict_delay / 2.0  # last action
            state[1, -1] = delay / 2.0
            state[2, -1] = min_remain_time / 2.0  # sec
            #state[3, -1] = (last_predict_bandwidth - predict_backhaul_bandwidth) / float(predict_backhaul_bandwidth)  # mega byte
            state[3, -1] = segment_idx / float(SEGMENT_MAX)  
            state[4, :A_DIM] = np.array(qoe_info) / request_num / float(np.max(TILE_SVC_BIT_RATE))  # predict qoe

            
            last_delay = delay
            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
            action_cumsum = np.cumsum(action_prob)
            action = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            entropy_record.append(a3c.compute_entropy(action_prob[0]))   
                                
            presupposed_bw = ACTION[action]*predict_backhaul_bandwidth            
            info_cluster = gamebased_allocating.gamebased_allocating(info_cluster,presupposed_bw)             
            time_stamp, users, request_queue, all_response, predict_backhaul_bandwidth, average_bandwidth, reward, delay= content_delivery(time_stamp,'other',info_cluster,users,request_queue,cache_table,user_num,request_num,net_env,all_response,average_bandwidth)                   
            reward = reward + 100*min(min_remain_time,0)
            print(action_prob)
 
            last_action = action    
            s_batch.append(state)
            action_vec = np.zeros(A_DIM)
            action_vec[action] = 1
            a_batch.append(action_vec)
            end_of_video = (not if_continue_playback(users))
            r_batch.append(reward)   

            if end_of_video:  #视频末尾
                del s_batch[:]
                del a_batch[:]
                del r_batch[:]
                entropy_record = []                                 
                break
        record_response(all_response,strategy,variable,user_num,bw_capacity,bw_fluctuation,cache_max,zipf,i)
                         

if __name__ == '__main__':
    main()
