#coding:utf-8
import os
import numpy as np
import math
import json
from user import env_user
import config as parameter

#USER_NUM = parameter.get_config('USER_NUM')
RRH_MAX = parameter.get_config('RRH_MAX')
video_names = parameter.get_config('video_names')
VIDEO_NUM = parameter.get_config('VIDEO_NUM')
SEGMENT_MAX = parameter.get_config('SEGMENT_MAX')
TILE_MAX = parameter.get_config('TILE_MAX')
tolerate = 0.0


def init_multi_users(user_num): #初始化用户
	users = []
	for user_idx in range(user_num):
		users.append(env_user(user_idx,user_idx % RRH_MAX + 1)) 
	return users

def init_request_queue(users,user_num,cache_table):#初始化请求队列
	request_queue = []
	request_queue = add_request(0,users,request_queue,0,user_num,cache_table)
	return request_queue


def add_request(time_stamp,users,request_queue,segment_delay,user_num,cache_table):
	for user_idx in range(user_num): #用户刚开始播放视频
		if users[user_idx].if_palyback(segment_delay):
			new_request = users[user_idx].send_request(time_stamp,cache_table)
			request_queue.append(new_request)
	return request_queue

def update_request_queue(time_stamp,users,request_queue,last_request,response,cache_table,user_num):
	segment_delay = response['segment_delay']
	#request_queue = add_request(time_stamp,users,request_queue,segment_delay,user_num,cache_table)
	#更新应答的用户请求
	request_queue.remove(last_request)
	response['qoe'] = users[response['user_idx']].evaluate_qoe(response)
	users[response['user_idx']].update_state(last_request,response)
	if users[response['user_idx']].status == 'PALYBACK':#继续播放
		next_request = users[response['user_idx']].send_request(time_stamp,cache_table) #应答用户的下一个请求
		request_queue.append(next_request)
	#更新排队时间和缓存情况
	#for idx,request in enumerate(request_queue):
		#request_queue[idx]['queueing_time'] = time_stamp - request['time_stamp']
		#request_queue[idx]['remain_time'] = request['deadline'] - request['queueing_time'] - tolerate
		#request = cache_match(request,cache_table)
	return users, request_queue, response

if __name__ == '__main__':
	users = init_multi_users()
	request_queue = init_request_queue(users)
	print(request_queue)
