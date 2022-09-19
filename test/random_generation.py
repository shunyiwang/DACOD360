#coding:utf-8
import os
import logging
import numpy as np
import pandas as pd
import random
import json
import config as parameter
import load_viewport
import shutil

VIDEO_NUM = parameter.get_config('VIDEO_NUM')
SLEEP_TIME = parameter.get_config('SLEEP_TIME')
VIEW_VIDEO_NUM = parameter.get_config('VIEW_VIDEO_NUM')
video_names = parameter.get_config('video_names')
#ZIPF_PARA = parameter.get_config('ZIPF_PARA')
SEGMENT_MAX = parameter.get_config('SEGMENT_MAX')
TILE_MAX = parameter.get_config('TILE_MAX')
TILE_BIT_RATE = parameter.get_config('TILE_BIT_RATE')
VOD_RATIO = parameter.get_config('VOD_RATIO')
VOD_NUM = parameter.get_config('VOD_NUM')
VOD_VIDEO = parameter.get_config('VOD_VIDEO')
LIVE_NUM = VIDEO_NUM - VOD_NUM

def view_user(user_num,zipf): #按照zipf分布生成观看视频数据
    folder = './user_view/'
    if os.path.exists(folder):
        shutil.rmtree(folder)
        os.makedirs(folder)
    else:
        os.makedirs(folder)
    popularity = (np.zeros((VOD_NUM,SEGMENT_MAX,TILE_MAX),dtype=int)).tolist()
    for user in range(user_num):
        view_video = []
        a = zipf #zipf parameter
        view_video_idx = VOD_NUM + 1      
        while view_video_idx > VOD_NUM-1: #or video_names[view_video_idx-1] in view_video:
            view_video_idx = np.random.zipf(a)
        view_video.append(video_names[view_video_idx-1])
        for segment_idx in range(SEGMENT_MAX):
            viewport = load_viewport.load_viewport(user,video_names[view_video_idx-1],segment_idx,0)
            for tile_idx in viewport:
                popularity[view_video_idx-1][segment_idx][tile_idx] +=1
    with open("cache/popularity.json","w") as file:
            json.dump(popularity,file)


def view_video(user_num,zipf,view_video_num = VIEW_VIDEO_NUM): #按照zipf分布生成观看视频数据\
    sample = np.random.exponential(SLEEP_TIME/float(user_num), size=user_num)
    time = np.cumsum(sample) 
    all_view_video = []
    folder = './user_view/'
    if os.path.exists(folder):
        shutil.rmtree(folder)
        os.makedirs(folder)
    else:
        os.makedirs(folder)
    for user in range(user_num):
        view_video = []
        sleep_time = time[user]        
        if user % int(1/VOD_RATIO) == 0: #直播
            for video in range(view_video_num):
                view_video_idx = LIVE_NUM + 1
                a = zipf #zipf parameter
                while view_video_idx > LIVE_NUM - 1: #or video_names[view_video_idx-1] in view_video:
                    view_video_idx = np.random.zipf(a)                
                view_video_idx = view_video_idx + VOD_NUM
                view_video.append(video_names[view_video_idx-1]) 
                data = {'view_video' : view_video, 'sleep_time':sleep_time} 
                all_view_video.append(data)                   
        else:
            for video in range(view_video_num):#点播
                view_video_idx = VOD_NUM + 1
                a = zipf #zipf parameter
                while view_video_idx > VOD_NUM - 1: #or video_names[view_video_idx-1] in view_video:
                    view_video_idx = np.random.zipf(a)
                view_video.append(video_names[view_video_idx-1])
                data = {'view_video' : view_video, 'sleep_time':sleep_time} 
                all_view_video.append(data)
    with open("user_view/view_video.json","w") as file:
        json.dump(all_view_video,file)

def view_sleep(user_num):
    sample = np.random.exponential(SLEEP_TIME/float(user_num), size=user_num)
    time = np.cumsum(sample)     
    for user in range(0,user_num):
        sleep_time = time[user]
        with open("user_view/view_video_"+str(user)+".json",'r') as fp:
            data = json.load(fp)
            view_video = data['view_video']
            if view_video[0] in VOD_VIDEO:
                data = {'view_video' : view_video, 'sleep_time':sleep_time}
            else:
                data = {'view_video' : view_video, 'sleep_time':0.0}                
        with open("user_view/view_video_"+str(user)+".json","w") as file:
            json.dump(data,file)

def video_size():
    video_size = []
    for i in range(SEGMENT_MAX+1):
        one_video_size = []
        config = np.random.normal(loc=1.0, scale=0.2, size=None)             
        for bitrate in TILE_BIT_RATE:
            one_video_size.append(config*bitrate/8.0)
        video_size.append(one_video_size)
    with open('video_size/video_size.json',"w") as file:
        json.dump(video_size,file)
        
        
def network_trace():
    backhaul_bw = []
    backgroud_bw = []    
    for i in range(60):
        #self.backhaul_bw.append(bw*bw_config)
        a = np.random.normal(loc = 0.0, scale = 2.0, size=None)
        while a < - 1.0 or a > 1.0:
            a = np.random.normal(loc = 0.0, scale = 2.0, size=None)
        backgroud_bw.append(a)
        backgroud_bw.append(-a)
    #random.shuffle(backgroud_bw)
    #for i in range(20):
        #backgroud_bw.append(0)
    for bw_fluctuation in range(10):
        one_bw = []
        for backgroud in backgroud_bw:
            simulated_bw = 1.0 - 0.1*bw_fluctuation*backgroud
            one_bw.append(simulated_bw)                        
        backhaul_bw.append(one_bw)      
    with open('bw_traces/backhual.json',"w") as file:
        json.dump(backhaul_bw,file)
''' 
def network_trace():
    backhaul_bw = []
    for bw_fluctuation in range(10):
        one_bw = []    
        for i in range(100):
            simulated_bw = np.random.normal(loc=0.0, scale=0.2*bw_fluctuation, size=None)
            while simulated_bw < -1.0 or simulated_bw > 1.0:
                simulated_bw = np.random.normal(loc=0.0, scale=0.2*bw_fluctuation, size=None)
            one_bw.append(1.0+simulated_bw)
            one_bw.append(1.0-simulated_bw)
        backhaul_bw.append(one_bw)
        print(one_bw,sum(one_bw))        
    with open('bw_traces/backhual.json',"w") as file:
        json.dump(backhaul_bw,file)

def network_trace():
    backhaul_bw = []
    backgroud_bw = []    
    for i in range(100):
        #self.backhaul_bw.append(bw*bw_config)
        a = np.random.normal(loc = 0.0, scale = 2.0, size=None)
        while a < - 1.0 or a > 1.0:
            a = np.random.normal(loc = 0.0, scale = 2.0, size=None)
        backgroud_bw.append(a)
        backgroud_bw.append(-a)
    for bw_fluctuation in range(10):
        one_bw = []
        for backgroud in backgroud_bw:
            simulated_bw = 1.0 - 0.1*bw_fluctuation*backgroud
            one_bw.append(simulated_bw)
        print(one_bw,sum(one_bw))                        
        backhaul_bw.append(one_bw)      
    with open('bw_traces/backhual.json',"w") as file:
        json.dump(backhaul_bw,file)

       
def network_trace():
    backhaul_bw = []
    bw = 1.0   
    for bw_fluctuation in range(10): 
        one_bw = []    
        for i in range(60):
            a = np.random.normal(loc = 0.0, scale = 0.02*bw_fluctuation, size=None)
            while a < -1.0 or a > 1.0:
                a = np.random.normal(loc = 0.0, scale = 0.02*bw_fluctuation, size=None)
            if a > 0:
                bw = bw*(1/(1.0-a))
            else:
                bw = bw*(1.0+a)
            one_bw.append(bw)
        normalize_factor = sum(one_bw)/65.0
        for idx,bw in enumerate(one_bw):
            one_bw[idx] = bw / normalize_factor
        print(one_bw,sum(one_bw))                        
        backhaul_bw.append(one_bw)      
    with open('bw_traces/backhual.json',"w") as file:
        json.dump(backhaul_bw,file)       
'''       
           
if __name__ == '__main__':
    all_viewport = load_viewport.load_all_viewport() 
    view_user(10000,1.5)
    view_video(1000,1.5)
    #iew_sleep(1000)
    video_size()
    network_trace()
