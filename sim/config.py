#coding:utf-8
import os
import logging
import numpy as np


def _init():
	#user state
	STARTING_TIME = 2 #sec
	SEGMENT_MAX = 60 #sec
	SEGMENT_LEN = 1
 
	alpha1 = 1.0
	alpha2 = 0.0
	alpha3 = 0.0
	delta = 0
	USERSTATUS=['SLEEP','PALYBACK','end']
	#viewport
	OLD_ROW_MAX = 10.0
	OLD_COL_MAX = 20.0 #row*col = 10*20
	NEW_ROW_MAX = 5.0
	NEW_COL_MAX = 10.0 #row*col = 3*4
	TILE_MAX = int(NEW_ROW_MAX * NEW_COL_MAX)
	fps = 30.0 #fps
	#多用户情况
	VIDEO_NUM = 10 #视频总数
	VIEW_VIDEO_NUM = 5 #假设每位用户观看10个视频
	#USER_ARRIVING = 1 #用户到达速率1人/s
	#USER_NUM = 20 #用户总数
	SLEEP_TIME = 0 #用户开始观看视频时间间隔
	video_names = ['coaster','coaster2','diving','drive','game','landscape','pacman','panel','ride','sport']#视频
	VOD_NUM =  7
	VOD_RATIO = 0.25
	VOD_VIDEO = video_names[:VOD_NUM]
	RRH_MAX = 5
	ZIPF_PARA = 1.5
	VIDEO_MAX = 1#实际观看视频数量
	TILE_BIT_RATE = [200.0,400.0,800.0,1600.0]  # Kbps SVC编码 默认基础层以及缓存 
	TILE_SVC_BIT_RATE = [TILE_BIT_RATE[0],TILE_BIT_RATE[0]+TILE_BIT_RATE[1],\
	TILE_BIT_RATE[0]+TILE_BIT_RATE[1]+TILE_BIT_RATE[2],TILE_BIT_RATE[0]+TILE_BIT_RATE[1]+TILE_BIT_RATE[2]+TILE_BIT_RATE[3]]
	#TILE_SVC_BIT_RATE = TILE_BIT_RATE
	RANDOM_SEED = 42
	PACKET_PAYLOAD_PORTION = 0.95
	POSITIONDICT = {'ac_rrh':4, 'bbu':3, 'nb_rrh':2, 'server':1}
	USER_MAX = 50
	QUALITY_MAX = len(TILE_BIT_RATE)
	CACHE_MAX = VOD_NUM*SEGMENT_MAX*TILE_MAX*max(TILE_BIT_RATE)/8.0 


	
	global _global_dict
	_global_dict = {'STARTING_TIME':STARTING_TIME, 'SEGMENT_MAX':SEGMENT_MAX, 'SEGMENT_LEN':SEGMENT_LEN, 'VOD_VIDEO':VOD_VIDEO,\
	'alpha1':alpha1, 'alpha2':alpha2, 'alpha3':alpha3, 'delta':delta, 'USERSTATUS':USERSTATUS, 'OLD_ROW_MAX':OLD_ROW_MAX, \
	'OLD_COL_MAX':OLD_COL_MAX, 'NEW_ROW_MAX':NEW_ROW_MAX, 'NEW_COL_MAX':NEW_COL_MAX, 'TILE_MAX':TILE_MAX, 'VOD_RATIO':VOD_RATIO, 'VOD_NUM':VOD_NUM,\
	'fps':fps, 'VIDEO_NUM':VIDEO_NUM, 'SLEEP_TIME':SLEEP_TIME, 'VIEW_VIDEO_NUM':VIEW_VIDEO_NUM, 'USER_MAX':USER_MAX,'CACHE_MAX':CACHE_MAX,\
	'video_names':video_names, 'RRH_MAX':RRH_MAX, 'ZIPF_PARA':ZIPF_PARA, 'VIDEO_MAX':VIDEO_MAX, 'TILE_BIT_RATE':TILE_BIT_RATE,\
	'RANDOM_SEED':RANDOM_SEED, 'PACKET_PAYLOAD_PORTION':PACKET_PAYLOAD_PORTION, 'POSITIONDICT':POSITIONDICT,'TILE_SVC_BIT_RATE':TILE_SVC_BIT_RATE,\
	'QUALITY_MAX':QUALITY_MAX}


def get_config(name):
	_init()
	try:
		return _global_dict[name]
	except KeyError:
		return defValue

if __name__ == '__main__':
	print(get_config('video_names'))
	