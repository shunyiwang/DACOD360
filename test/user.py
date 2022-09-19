#coding:utf-8
import os
import logging
import numpy as np
import multiprocessing as mp     #multiprocessing包是Python中的多进程管理包
import math
import json
import config as parameter


import load_viewport as load_viewport
import load_userview as load_userview
#from cache_replace import cache_update
#COOKED_CACHE_FOLDER = './cache_table/'


STARTING_TIME = parameter.get_config('STARTING_TIME')
SEGMENT_MAX = parameter.get_config('SEGMENT_MAX')
SEGMENT_LEN = parameter.get_config('SEGMENT_LEN')
alpha1 = parameter.get_config('alpha1')
alpha2 = parameter.get_config('alpha2')
alpha3 = parameter.get_config('alpha3')
delta = parameter.get_config('delta')
USERSTATUS = parameter.get_config('USERSTATUS')
video_names = parameter.get_config('video_names')
VIDEO_MAX = parameter.get_config('VIDEO_MAX')
TILE_SVC_BIT_RATE = parameter.get_config('TILE_SVC_BIT_RATE')
RRH_MAX = parameter.get_config('RRH_MAX')
QUALITY_MAX = parameter.get_config('QUALITY_MAX')
VOD_VIDEO = parameter.get_config('VOD_VIDEO')
#BUFFER_MAX = 1.99



class env_user:
    """docstring for env_user"""
    def __init__(self, user_idx,rrh,opt):
        self.user_idx = user_idx
        self.rrh = rrh
        self.playbacktime = 0
        self.buffer = 0
        self.starting_time = 0
        self.video_idx = 0
        self.video, self.sleep_time = load_userview.load_userview(self.user_idx, self.video_idx)
        self.persist_sleep_time = 0
        self.status = USERSTATUS[0]
        self.last_qoe_1 = 0
        self.last_quality = 0
        self.segment_idx = 0
        self.deadline = STARTING_TIME
        self.opt = opt
        if self.video in VOD_VIDEO:    
            self.video_type = 'VOD'
        else:
            self.video_type = 'LIVE'
        with open('video_size/video_size.json','r') as fp:
            self.video_size = json.load(fp)
            

    def predict_viewport(self):#预测视野 可换成复杂的预测方法
        if self.segment_idx == 0:
            #viewport,viewpoint = load_viewport.get_central_viewport()
            viewport = load_viewport.load_viewport(self.user_idx, self.video, self.segment_idx,0)      
        else:
            #viewport,viewpoint = load_viewport.load_viewport(self.user_idx, self.video, self.segment_idx, self.deadline)
            if self.opt:
                viewport = load_viewport.load_viewport(self.user_idx, self.video, self.segment_idx, 0)
            else:
                #viewport = load_viewport.load_viewport(self.user_idx, self.video, max(self.segment_idx-1,0), 0)
                viewport = load_viewport.load_viewport(self.user_idx, self.video, int(math.ceil(self.segment_idx-max(self.buffer,0))),0)
                #viewport = load_viewport.load_viewport(self.user_idx, self.video, self.segment_idx, 0)
        return viewport

    def get_viewport(self):#读取实际视野文件
        #print(self.user_idx+1, self.video, self.segment_idx)
        viewport = load_viewport.load_viewport(self.user_idx, self.video, self.segment_idx, 0)
        return viewport

    def update_cache_quality(self,viewport,cache_table):
        cache_quality_list = []
        for tile_idx in viewport:
            cache_quality_list.append(cache_table[video_names.index(self.video)][self.segment_idx][tile_idx])          
        cache_quality = np.mean(cache_quality_list)
        return cache_quality

    def predict_qoe(self):#qoe预测
        qoe_info = []
        for quality,bitrate in enumerate(TILE_SVC_BIT_RATE):
            qoe_1 = bitrate
            if self.segment_idx == 0:
                qoe_2 = 0.0
            else:
                qoe_2 = abs(qoe_1 - self.last_qoe_1)
            predict_qoe = max(alpha1*qoe_1 - alpha2*qoe_2,0)                                  
            qoe_info.append(predict_qoe)
        return qoe_info

    def send_request(self,time_stamp,cache_table): #发送request
        viewport = self.predict_viewport()  
        #cache_quality = self.update_cache_quality(viewport,cache_table)
        qoe_info = self.predict_qoe()
        request = {'time_stamp':time_stamp, 'user_idx':self.user_idx, 'associative_rrh':self.rrh, 'video':self.video, 'qoe_info':qoe_info, 'video_type':self.video_type,\
        'segment_idx':self.segment_idx, 'deadline':self.deadline, 'buffer':self.buffer, 'viewport':viewport, 'tile_size':self.video_size[self.segment_idx], 'cluster_mark':0,\
        'cache_quality':0,'queueing_time':0,'remain_time':self.deadline,'quality':0,'last_qoe_1':self.last_qoe_1, 'last_quality':self.last_quality}
        #print(request)
        return request


    def if_miss_deadline(self,response):#判断是否真的错过了deadline
        remain_time = response['deadline'] - response['delay']
        if remain_time >= 0:
            return False
        else:
            return True

    def evaluate_qoe(self,response):#qoe评价：质量-时空抖动
        viewport_qoe = []
        real_viewport = self.get_viewport()
        predict_viewport = response['viewport']
        quality = response['quality']        
        for tile in real_viewport:
            if tile in predict_viewport:
                viewport_qoe.append(TILE_SVC_BIT_RATE[quality]) #真实的视野和预测视野偏差
            else:
                viewport_qoe.append(TILE_SVC_BIT_RATE[0])
        qoe_1 = np.mean(viewport_qoe)
        if self.segment_idx == 0:
            qoe_2 = 0
        else:
            qoe_2 = abs(qoe_1 - self.last_qoe_1)
        qoe_3 = (np.std(viewport_qoe))
        real_qoe = alpha1*qoe_1 - alpha2*qoe_2 - alpha3*qoe_3
        if self.if_miss_deadline(response):
            real_qoe = 0                 
            qoe_1 = 0
            qoe_2 = self.last_qoe_1 
        else:
            self.last_qoe_1 = qoe_1                             
        #print(quality,real_qoe,qoe_1,qoe_2,qoe_3)
        return max(real_qoe,0), qoe_1, [qoe_1,qoe_2,qoe_3]

    def update_state(self,request,response):#用户状态更新
        #print(self.user_idx, self.video, self.segment_idx, self.buffer, self.persist_sleep_time)
        if self.segment_idx >= SEGMENT_MAX-1: #new video
            if self.video_idx < VIDEO_MAX - 1:
            #if self.all_segment < VIDEO_MAX*SEGMENT_MAX - 1:
                self.video_idx = self.video_idx + 1        
                self.video, self.sleep_time, _ = load_userview.load_userview(self.user_idx, self.video_idx)
                self.segment_idx = 0
                self.last_qoe_1 = 0
                self.deadline = STARTING_TIME
                self.persist_sleep_time = 0
                self.starting_time = 0
                self.status = USERSTATUS[0]
                self.buffer = 0
                self.playbacktime = 0
                self.last_quality = 0
            else:
                self.status = USERSTATUS[2] #已经播放完所有视频
                return
        else:
            self.last_quality = response['quality']
            delay = response['delay']
            if self.starting_time < STARTING_TIME: #缓冲中
                self.buffer = self.buffer + SEGMENT_LEN
                self.playbacktime = max((self.starting_time - STARTING_TIME),0)
                self.deadline = self.buffer + (STARTING_TIME - self.starting_time)
                self.starting_time = self.starting_time + delay
            else:
                self.playbacktime = self.playbacktime + delay
                self.buffer = self.buffer + SEGMENT_LEN - delay
                self.deadline = self.buffer - delta*SEGMENT_LEN
                #self.deadline = 0
            #if self.buffer < 0:#miss deadline
                #self.segment_idx = int(math.ceil(self.playbacktime))
                #if self.segment_idx >= SEGMENT_MAX-1:
                    #self.update_state(request,response)
                #else:
                #self.buffer = self.deadline = 0
            #if self.buffer > BUFFER_MAX and self.quality == QUALITY_MAX:
                #self.buffer = self.deadline = BUFFER_MAX
      #elif self.buffer > 5:
          #self.starting_time = 0
          #self.status = USERSTATUS[0]
            self.segment_idx += 1


    def if_palyback(self,segment_delay):
        if self.status == "SLEEP":
            if self.sleep_time <= self.persist_sleep_time:
                self.status = USERSTATUS[1]
                return True
            else:
                self.persist_sleep_time = self.persist_sleep_time + segment_delay
                return False
        else:
            return False


if __name__ == '__main__':
    env_user = env_user(1)
    request = env_user.send_request(1)
    print(request)


