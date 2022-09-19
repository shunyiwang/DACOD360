#coding:utf-8
import os
import numpy as np
import math
import shutil
import json
import config as parameter
import decision
import random_generation
import result_analyse
import matplotlib.pyplot as plt
import pandas as pd
import config as parameter
import shutil
import load_viewport
import load_userview
import rl_decision
MARK = ['-*','-o','-v','-s','-p','-x']
#VARIABLE = ['User arrival rate','Bandwidth','Cache capacity','Zipf parameter']
VARIABLE = ['Bandwidth capacity','Cache capacity','User num','Bandwidth fluctuation']
#par = pd.Series([1,2,4,8,16])
#par = pd.Series([1])
#Zipf parameter'
ZIPF = [1.01,1.5,2.0,2.5,3.0]
BW_CAPACITY = [300,400,500,600,700,800,900]
#BW_CAPACITY = [300]
BW_FLUCTUATION = [0,10,20,30,40,50,60]
USER_NUM = [100,150,200,250,300,350,400]
ROUND = 1
#,'baseline','offline_optimal'
CACHE = [10,13,16,19,22,25,28]
STRATEGY = ['rl_game','proposed','temporal_greedy','spatio_greedy','greedy','no_cache']
#STRATEGY = ['temporal_greedy','greedy']
SLEEP_TIME = parameter.get_config('SLEEP_TIME')


def running(variable):
    bw_capacity = BW_CAPACITY[2]
    bw_fluctuation = BW_FLUCTUATION[3]
    zipf = ZIPF[1]
    user_num = USER_NUM[2]
    cache_max = CACHE[1]
    if os.path.exists('./response/'+str(variable)) == True: 
        shutil.rmtree('./response/'+str(variable))
    #random_generation.view_user(user_num=10000, zipf = zipf)   
    #random_generation.view_video(user_num=user_num, zipf = zipf)
    #random_generation.view_sleep(user_num=user_num)
    #random_generation.video_size()      
    if variable == 'Bandwidth capacity':
        for bw_capacity in BW_CAPACITY:
            for i in range(ROUND):
                #random_generation.view_sleep(user_num=user_num)
                for strategy in STRATEGY:
                    if strategy == 'rl_game':
                        rl_decision.rl_decision(strategy,variable,user_num,bw_capacity,bw_fluctuation,cache_max,zipf,i)
                    else:             
                        decision.decision(strategy,variable,user_num,bw_capacity,bw_fluctuation,cache_max,zipf,i)
    elif variable == 'Bandwidth fluctuation':
        for bw_fluctuation in BW_FLUCTUATION:
            for i in range(ROUND):
                #random_generation.view_sleep(user_num=user_num)
                for strategy in STRATEGY:                      
                    runing_time = decision.decision(strategy,variable,user_num,bw_capacity,bw_fluctuation,cache_max,zipf,i)
    elif variable == 'User num':
        for user_num in USER_NUM:
            for i in range(ROUND):
                #random_generation.view_user(user_num=10000, zipf = zipf)   
                #random_generation.view_video(user_num=user_num, zipf = zipf)
                #random_generation.view_sleep(user_num=user_num)
                for strategy in STRATEGY:
                    runing_time = decision.decision(strategy,variable,user_num,bw_capacity,bw_fluctuation,cache_max,zipf,i)
    elif variable == 'Cache capacity':
        for cache_max in CACHE:
            for i in range(ROUND):
                #random_generation.view_sleep(user_num=user_num)
                for strategy in STRATEGY:
                    runing_time = decision.decision(strategy,variable,user_num,bw_capacity,bw_fluctuation,cache_max,zipf,i)
    elif variable == 'Zipf parameter':
        for zipf in ZIPF:
            for i in range(ROUND):
                #random_generation.view_user(user_num=10000, zipf = zipf)       
                #random_generation.view_video(user_num=user_num, zipf = zipf)
                #random_generation.view_sleep(user_num=user_num)
                for strategy in STRATEGY:
                    runing_time = decision.decision(strategy,variable,user_num,bw_capacity,bw_fluctuation,cache_max,zipf,i)	


def plot_result(all_var,all_multi_qoe,all_multi_jain,all_multi_quality,all_multi_miss,all_multi_hit,all_multi_remain,all_multi_traffic,variable):
	plt.subplot(231)
	for idx,strategy in enumerate(STRATEGY): 
		plt.plot(all_var,all_multi_qoe[idx],MARK[idx])
	plt.legend(STRATEGY)
	plt.xlabel(variable)
	plt.ylabel('QoE') 
   
	
	plt.subplot(232)
	for idx,strategy in enumerate(STRATEGY):
		plt.plot(all_var,all_multi_jain[idx],MARK[idx])
	plt.legend(STRATEGY)
	plt.xlabel(variable)  
	plt.ylabel("Jain's fairness index") 

	plt.subplot(233)
	for idx,strategy in enumerate(STRATEGY):
		plt.plot(all_var,all_multi_quality[idx],MARK[idx])
	plt.legend(STRATEGY)
	plt.xlabel(variable)  
	plt.ylabel('Quality')  
	#plt.title('Quality v.s.'+variable)

	plt.subplot(234)
	for idx,strategy in enumerate(STRATEGY):
		plt.plot(all_var,all_multi_miss[idx],MARK[idx])
	plt.legend(STRATEGY)
	plt.xlabel(variable)  
	plt.ylabel('Missdeadline_ratio')  
	#plt.title('Missdeadline_ratio v.s.'+variable)
	
	plt.subplot(235)
	for idx,strategy in enumerate(STRATEGY):
		if strategy != 'no_cache':
			plt.plot(all_var,all_multi_traffic[idx],MARK[idx])
	plt.legend(STRATEGY)
	plt.xlabel(variable)  
	plt.ylabel("Traffic_load_reduction") 
	'''
	plt.subplot(235)
	for idx,strategy in enumerate(STRATEGY):
		plt.plot(all_var,all_multi_hit[idx],MARK[idx])
	plt.legend(STRATEGY)
	plt.xlabel(variable)  
	plt.ylabel('Cachehit_ratio')  
	'''
	plt.subplot(236)
	for idx,strategy in enumerate(STRATEGY):
		plt.plot(all_var,all_multi_remain[idx],MARK[idx])
	plt.legend(STRATEGY)
	plt.xlabel(variable)  
	plt.ylabel('Reman_time')  
	#plt.title('Missdeadline_ratio v.s.'+variable)
	plt.show()



def analyse(variable):
	#print(runing_time)
	#variable = 'Bandwidth'
	if os.path.exists('./result/'+str(variable)):
		shutil.rmtree('./result/'+str(variable))
		os.makedirs('./result/'+str(variable)) 
	else:
		os.makedirs('./result/'+str(variable))
	if os.path.exists('./csv.'):
		shutil.rmtree('./csv.')
		os.makedirs('./csv.') 
	else:
		os.makedirs('./csv.')     
	all_multi_qoe = []
	all_multi_jain = []
	all_multi_quality = []
	all_multi_miss = []
	all_multi_hit = []
	all_multi_remain = []
	all_multi_traffic = [] 
	for strategy in STRATEGY:
		var,multi_qoe,multi_jain,multi_quality,multi_miss,multi_hit,multi_remain,multi_traffic = result_analyse.result_analyse(strategy,variable)
		all_multi_qoe.append(multi_qoe)
		all_multi_jain.append(multi_jain)
		all_multi_quality.append(multi_quality)
		all_multi_miss.append(multi_miss)
		all_multi_hit.append(multi_hit)
		all_multi_remain.append(multi_remain)
		all_multi_traffic.append(multi_traffic)   
	#all_var.sort()
	print(var,all_multi_qoe,all_multi_jain,all_multi_quality,all_multi_miss,all_multi_hit)
	plot_result(var,all_multi_qoe,all_multi_jain,all_multi_quality,all_multi_miss,all_multi_hit,all_multi_remain,all_multi_traffic,variable)


def main():
	all_viewport = load_viewport.load_all_viewport() 
	all_userview = load_userview.load_all_userview()
	#for variable in VARIABLE:
		#running(variable)
	running('Bandwidth capacity')
	analyse('Bandwidth capacity')


if __name__ == '__main__':
	main()