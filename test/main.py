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
import scipy.stats as stats
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
color = ['#4695C0','#FB943C','#49B265','#D62828','#AF87CE','#B2967D']
MARK = ['-*','-o','-v','-s','-p','-x']
#VARIABLE = ['User arrival rate','Bandwidth','Cache capacity','Zipf parameter']
VARIABLE = ['Average Bandwidth (Mbps)','Cache Capacity (%)','User Number','Bandwidth Fluctuation (%)']
#par = pd.Series([1,2,4,8,16])
#par = pd.Series([1])
#Zipf parameter'
#ZIPF = [1.01,1.5,2.0,2.5,3.0]
BW_CAPACITY = [100,150,200,250,300,350,400]
BW_FLUCTUATION = [20,30,40,50,60,70,80]
USER_NUM = [200,250,300,350,400,450,500]
ZIPF = [1.5,4.5,7.5]
ROUND = 1
#,'baseline','offline_optimal'
CACHE = [13,15,17,19,21,23,25]
TRACE_MAX = 50
STRATEGY = ['Offline_OPT','DACOD360','DRL_ONLY','CBG_ONLY','TJCD360','NO_CACHE']
SLEEP_TIME = parameter.get_config('SLEEP_TIME')


def running(variable):
    bw_capacity = BW_CAPACITY[2]
    bw_fluctuation = BW_FLUCTUATION[3]
    zipf = ZIPF[1]
    user_num = USER_NUM[2]
    cache_max = CACHE[1]
    trace_idx = 0
    if os.path.exists('./response/'+str(variable)) == True: 
        shutil.rmtree('./response/'+str(variable))
    random_generation.view_user(user_num=10000, zipf = zipf)   
    random_generation.view_video(user_num=1000, zipf = zipf)     
    if variable == 'Average Bandwidth (Mbps)':
        for bw_capacity in BW_CAPACITY:
            for i in range(ROUND):
                for strategy in STRATEGY:
                    if strategy == 'DACOD360' or strategy == 'DRL_ONLY' or strategy == 'NO_CACHE':
                        rl_decision.rl_decision(strategy,variable,user_num,bw_capacity,bw_fluctuation,cache_max,trace_idx,zipf,i)
                    else:             
                        decision.decision(strategy,variable,user_num,bw_capacity,bw_fluctuation,cache_max,trace_idx,zipf,i)
    elif variable == 'Bandwidth Fluctuation (%)':
        for bw_fluctuation in BW_FLUCTUATION:
            for i in range(ROUND):
                for strategy in STRATEGY:                      
                    if strategy == 'DACOD360' or strategy == 'DRL_ONLY' or strategy == 'NO_CACHE':
                        rl_decision.rl_decision(strategy,variable,user_num,bw_capacity,bw_fluctuation,cache_max,trace_idx,zipf,i)
                    else:             
                        decision.decision(strategy,variable,user_num,bw_capacity,bw_fluctuation,cache_max,trace_idx,zipf,i)
    elif variable == 'qoe':
            for i in range(ROUND):
                for strategy in STRATEGY:                      
                    if strategy == 'DACOD360' or strategy == 'DRL_ONLY' or strategy == 'NO_CACHE':
                        rl_decision.rl_decision(strategy,variable,user_num,bw_capacity,bw_fluctuation,cache_max,trace_idx,zipf,i)
                    else:             
                        decision.decision(strategy,variable,user_num,bw_capacity,bw_fluctuation,cache_max,trace_idx,zipf,i)     
                          
    elif variable == 'User Number':
        for user_num in USER_NUM:
            for i in range(ROUND):
                for strategy in STRATEGY:
                    if strategy == 'DACOD360' or strategy == 'DRL_ONLY' or strategy == 'NO_CACHE':
                        rl_decision.rl_decision(strategy,variable,user_num,bw_capacity,bw_fluctuation,cache_max,trace_idx,zipf,i)
                    else:             
                        decision.decision(strategy,variable,user_num,bw_capacity,bw_fluctuation,cache_max,trace_idx,zipf,i)
    elif variable == 'Cache Capacity (%)':
        for cache_max in CACHE:
            for i in range(ROUND):
                for strategy in STRATEGY:
                    if strategy == 'DACOD360' or strategy == 'DRL_ONLY' or strategy == 'NO_CACHE':
                        rl_decision.rl_decision(strategy,variable,user_num,bw_capacity,bw_fluctuation,cache_max,trace_idx,zipf,i)
                    else:             
                        decision.decision(strategy,variable,user_num,bw_capacity,bw_fluctuation,cache_max,trace_idx,zipf,i)
    elif variable == 'multi_network':
        for trace_idx in range(TRACE_MAX):
            for i in range(ROUND):
                for strategy in STRATEGY:
                    if strategy == 'DACOD360' or strategy == 'DRL_ONLY' or strategy == 'NO_CACHE':
                        rl_decision.rl_decision(strategy,variable,user_num,bw_capacity,bw_fluctuation,cache_max,trace_idx,zipf,i)
                    else:
                        decision.decision(strategy,variable,user_num,bw_capacity,bw_fluctuation,cache_max,trace_idx,zipf,i)
    elif variable == 'zipf':
        for zipf in ZIPF:
            for i in range(ROUND):
                random_generation.view_user(user_num=10000, zipf = zipf)       
                random_generation.view_video(user_num=1000, zipf = zipf)
                for strategy in STRATEGY:
                    if strategy == 'DACOD360' or strategy == 'DRL_ONLY' or strategy == 'NO_CACHE':
                        rl_decision.rl_decision(strategy,variable,user_num,bw_capacity,bw_fluctuation,cache_max,trace_idx,zipf,i)
                    else:
                        decision.decision(strategy,variable,user_num,bw_capacity,bw_fluctuation,cache_max,trace_idx,zipf,i)                        
                        
                        
                        
def plot_cdf(variable):
    bar_width = 0.13
    x = variable.split("(")
    plt.rc('font',family='Times New Roman')
    path = './result/'+str(variable)+'.json'
    with open(path,"r") as file:
        data = json.load(file)
    all_var = np.array(data['var'])
    all_multi_qoe = data['all_multi_qoe']
    all_multi_quality = data['all_multi_quality']
    all_multi_miss = data['all_multi_miss']
    all_multi_traffic = data['all_multi_traffic']
    value = [all_multi_qoe,all_multi_quality,all_multi_miss,all_multi_traffic]
    ylabels = ['Average QoE','Average Bitrate (Mbps)','Miss-deadline Ratio (%)','Backhaul Traffic Reduction (GB)']
    fig = plt.figure(dpi=100,figsize=(6,6))       
    nums = ['(a) ','(b) ','(c) ','(d) ']
    axes = fig.subplots(nrows=2, ncols=2)
    for i,ax in enumerate(fig.axes):
        if i == 0:
            ax.set_xticks([4.5,5.5,6.5])
        ax.grid(ls = "--", color = "#4E616C") 
        ax.set_ylabel('CDF',fontsize=10)
        #y = ylabels[i].split("(")
        y = ylabels[i]
        ax.set_title(nums[i] + y, y=-0.25, fontsize=12)
        
        for idx,strategy in enumerate(STRATEGY):
            if i == 3 and strategy == 'NO_CACHE':
                continue
            #if i == 2 and (strategy == 'Offline_OPT'):
                #continue
            #values, base = np.histogram(sorted(value[i][idx]),bins=TRACE_MAX)
            #cumulative = np.cumsum(values)
            #ax.plot(base[:-1], cumulative/float(TRACE_MAX))
            res = stats.relfreq(sorted(value[i][idx]), numbins=TRACE_MAX)
            xx = res.lowerlimit + np.linspace(0, res.binsize*res.frequency.size,res.frequency.size)
            xx[0] = max(xx[0],0)
            yy = np.cumsum(res.frequency)
            ax.plot(xx,yy)
        if i == 2:
            res = stats.relfreq(sorted(value[i][1]), numbins=TRACE_MAX)
            xx = res.lowerlimit + np.linspace(0, res.binsize*res.frequency.size,res.frequency.size)
            xx[0] = max(xx[0],0)
            yy = np.cumsum(res.frequency)
            ax.set_xlim(-1,28)
            ax.plot(xx,yy,color = 'tab:orange')     
             
             
            #子图
            axins = inset_axes(ax, width="40%", height="30%", loc='lower right',
                       bbox_to_anchor=(0, 0.12, 1, 1), 
                       bbox_transform=ax.transAxes)
            axins.grid(ls = "--", color = "#4E616C") 
            for idx,strategy in enumerate(STRATEGY):
                res = stats.relfreq(sorted(value[i][idx]), numbins=TRACE_MAX)
                xx = res.lowerlimit + np.linspace(0, res.binsize*res.frequency.size,res.frequency.size)
                xx[0] = max(xx[0],0)
                yy = np.cumsum(res.frequency)
                axins.plot(xx,yy)
            res = stats.relfreq(sorted(value[i][1]), numbins=TRACE_MAX)
            xx = res.lowerlimit + np.linspace(0, res.binsize*res.frequency.size,res.frequency.size)
            xx[0] = max(xx[0],0)
            yy = np.cumsum(res.frequency)
            axins.plot(xx,yy,color = 'tab:orange') 
            xlim0 = -0.1
            xlim1 = 3.5
            ylim0 = 0.7
            ylim1 = 1.05
            
            
            
            # 调整子坐标系的显示范围
            axins.set_xlim(xlim0, xlim1)
            axins.set_ylim(ylim0, ylim1)
            
            
            # 原图中画方框
            tx0 = xlim0
            tx1 = xlim1
            ty0 = ylim0
            ty1 = ylim1
            sx = [tx0,tx1,tx1,tx0,tx0]
            sy = [ty0,ty0,ty1,ty1,ty0]
            ax.plot(sx,sy,"black")
            
            # 画两条线
            xy = (xlim0,ylim0)
            xy2 = (xlim0,ylim1)
            con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
                    axesA=axins,axesB=ax)
            axins.add_artist(con)
            
            xy = (xlim1,ylim0)
            xy2 = (xlim1,ylim1)
            con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
                    axesA=axins,axesB=ax)
            axins.add_artist(con)
    fig.legend(labels=STRATEGY, loc = 'upper center', ncol = 3) # 图例的位置，bbox_to_anchor=(0.5, 0.92),
    #plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.savefig('./figure/'+x[0]+'.pdf',dpi=600,format='pdf')
    plt.show()  

def plot_bar(variable):
    bar_width = 0.25
    x = variable.split("(")
    plt.rc('font',family='Times New Roman')
    path = './result/'+str(variable)+'.json'
    with open(path,"r") as file:
        data = json.load(file)
    all_var = np.array(data['var'])
    all_multi_qoe = data['all_multi_qoe']
    all_multi_quality = data['all_multi_quality']
    all_multi_miss = data['all_multi_miss']
    all_multi_traffic = data['all_multi_traffic']
    value = [all_multi_qoe,all_multi_quality,all_multi_miss,all_multi_traffic]
    ylabels = ['Average QoE','Average Bitrate (Mbps)','Miss-deadline Ratio (%)','Backhaul Traffic Reduction (GB)']
    tick_label = ['trace_1','trace_2','trace_3']
    fig = plt.figure(dpi=100,figsize=(6,6))       
    nums = ['(a) ','(b) ','(c) ','(d) ']
    axes = fig.subplots(nrows=2, ncols=2)
    for i,ax in enumerate(fig.axes):
        #ax.grid(ls = "--", color = "#4E616C")
        ax.set_xticks([]) 
        #ax.tick_params(axis='x',tick_label)
        #ax.set_xlabel(variable,fontsize=10)
        ax.set_ylabel(ylabels[i],fontsize=10)
        y = ylabels[i].split("(")
        ax.set_title(nums[i] + y[0], y=-0.15, fontsize=12)
        for idx,strategy in enumerate(STRATEGY):
            ax.bar(all_var+bar_width*idx,value[i][idx],bar_width)
    fig.legend(labels=STRATEGY, loc = 'upper center', ncol = 3) # 图例的位置，bbox_to_anchor=(0.5, 0.92),
    #plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    #fig.savefig('./figure/'+x[0]+'.pdf',dpi=600,format='pdf')
    plt.show()  

def plot_bar2(bw_idx):
    bar_width = 0.13
    plt.rc('font',family='Times New Roman')
    path = './result/'+str('Average Bandwidth (Mbps)')+'.json'
    with open(path,"r") as file:
        data = json.load(file)
    qoe_0 = [i[bw_idx] for i in data['qoe_0']]
    qoe_1 = [i[bw_idx] for i in data['qoe_1']]
    qoe_2 = [i[bw_idx] for i in data['qoe_2']]
    barx = np.arange(3)
    value = [qoe_0,qoe_1,qoe_2]
    value = tuple(zip(*value))
    labels = STRATEGY
    ylabels = 'QoE'
    tick_label = ['QoE_1','QoE_2','QoE_3']
    fig = plt.figure(dpi=100,figsize=(6,6)) 
    for idx,strategy in enumerate(STRATEGY):
        plt.bar(barx+bar_width*idx,value[idx],bar_width)     
    fig.legend(labels=STRATEGY, loc = 'upper center', ncol = 3) # 图例的位置，bbox_to_anchor=(0.5, 0.92),
    #plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.savefig('./figure/qoe.pdf',dpi=600,format='pdf')
    plt.show()   


def plot_bar3():
    bar_width = 0.13
    plt.rc('font',family='Times New Roman')
    path = './result/qoe.json'
    with open(path,"r") as file:
        data = json.load(file)
    value = [data['all_multi_qoe']]
    path = './result/zipf.json'
    with open(path,"r") as file:
        data = json.load(file)
    value.append(data['all_multi_qoe'])
    #value = tuple(zip(*value))
    labels = STRATEGY
    ylabels = 'QoE'
    barx = np.arange(3)
    fig = plt.figure(dpi=100,figsize=(6,6)) 
    tick_label = [('(1.0,0.5,0.5)','(0.5,1.0,0.5)','(0.5,0.5,1.0)'),(1.5,2.5,3.5)]
    tile = ['(a) Different QoE Parameter','(b) Different Zipf Parameter']
    pos_list = np.arange(len(tick_label[0])) + bar_width*2.5
    axes = fig.subplots(nrows=2, ncols=1)
    for i,ax in enumerate(fig.axes):
        for idx,strategy in enumerate(STRATEGY):
            ax.grid(ls = "--", color = "#4E616C") 
            ax.xaxis.set_major_locator(ticker.FixedLocator((pos_list)))
            ax.xaxis.set_major_formatter(ticker.FixedFormatter((tick_label[i])))
            ax.bar(barx+bar_width*idx,value[i][idx],bar_width,color = color[idx]) 
            #ax.tick_params(axis='x',tick_label)  
            #ax.set_xlabel(variable,fontsize=10)
            ax.set_title(tile[i], y=-0.25, fontsize=12)
            ax.set_ylabel(ylabels,fontsize=10)
            ax.set_ylim(3,6)
        #ax.legend(labels=STRATEGY, loc = 'upper center', ncol = 3) # 图例的位置，bbox_to_anchor=(0.5, 0.92),
        #plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    fig.legend(labels=STRATEGY, loc = 'upper center', ncol = 3) # 图例的位置，bbox_to_anchor=(0.5, 0.92),
    fig.savefig('./figure/qoe.pdf',dpi=600,format='pdf')
    plt.show()  








def plot_psnr():
    plt.rc('font',family='Times New Roman')
    interval = [0.5,1,2,3,4,5,6,7,8,9]
    precision = [[0.825790287,0.78908845,0.723076296,0.676704893,0.651140736,0.647777187,0.646803159,0.639130516,0.630933407,0.62503889],\
    [0.793548345,0.73226247,0.647205244,0.602888886,0.561582558,0.564151989,0.557710125,0.562471375,0.564786934,0.560172329],\
    [0.750158525,0.710177922,0.597883968,0.526336284,0.487823655,0.453386682,0.438556779,0.434877973,0.431496902,0.419164901]]

    precision = (np.array(precision)*100.0).tolist()
    
    bitrate = [3000,2000,1000,500,200,100,50]
    psnr = [[36.913921,34.324972,30.302756,26.835951,22.741618,20.983481,20.481433],\
    [38.54975,36.021319,32.553465,29.63886,25.547512,21.920339,20.847943],\
    [41.71549,39.816474,37.160575,34.761156,32.075365,30.09317,27.421803]]
 
    xvalue = [interval,bitrate]
    yvalue = [precision,psnr]

    ylabels = ['Prediction Precision (%)','PSNR']
    xlabels = ['Prediction Interval (s)','Bitrate (kbps)']
    title = ['Precision vs Interval','PSNR vs Bitrate']
    labels = ['Coaster','Game','Landscape']
    fig = plt.figure(dpi=100,figsize=(6,6))       
    nums = ['(a) ','(b) ']
    axes = fig.subplots(nrows=2, ncols=2)
    for i,ax in enumerate(fig.axes):
        if i > 1:
            break
        ax.grid(ls = "--", color = "#4E616C") 
        ax.tick_params(labelsize=10)
        ax.set_xlabel(xlabels[i],fontsize=10)
        ax.set_ylabel(ylabels[i],fontsize=10)
        ax.set_title(nums[i] + title[i], y=-0.38, fontsize=12)
        for idx,label in enumerate(labels): 
            ax.plot(xvalue[i],yvalue[i][idx],linewidth=1)            
    fig.legend(labels=labels, loc = (0.27,0.9), ncol = 3) # 图例的位置，bbox_to_anchor=(0.5, 0.92),
    #plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.48)
    fig.savefig('./figure/psnr.pdf',dpi=600,format='pdf')
    plt.show()




def plot_result(variable):
    x = variable.split("(")
    plt.rc('font',family='Times New Roman')
    path = './result/'+str(variable)+'.json'
    with open(path,"r") as file:
        data = json.load(file)
    all_var = data['var']
    all_multi_qoe = data['all_multi_qoe']
    all_multi_quality = data['all_multi_quality']
    all_multi_miss = data['all_multi_miss']
    all_multi_traffic = data['all_multi_jain']
    value = [all_multi_qoe,all_multi_quality,all_multi_miss,all_multi_traffic]
    ylabels = ['Average QoE','Average Bitrate (Mbps)','Miss-deadline Ratio (%)','Backhaul Traffic Reduction (GB)']
    nums = ['(a) ','(b) ','(c) ','(d) ']
    fig = plt.figure(dpi=100,figsize=(6,6))
    axes = fig.subplots(nrows=2, ncols=2)
    for i,ax in enumerate(fig.axes):
        ax.grid(ls = "--", color = "#4E616C") 
        ax.tick_params(labelsize=10)
        ax.set_xlabel(variable,fontsize=10)
        ax.set_ylabel(ylabels[i],fontsize=10)
        y = ylabels[i].split("(")
        ax.set_title(nums[i] + y[0], y=-0.38, fontsize=12)
        for idx,strategy in enumerate(STRATEGY):
            ax.plot(all_var,value[i][idx],MARK[idx],linewidth=1)
    #lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(labels=STRATEGY, loc = 'upper center', ncol = 3) # 图例的位置，bbox_to_anchor=(0.5, 0.92),
    #plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.48)
    fig.savefig('./figure/'+x[0]+'.pdf',dpi=600,format='pdf')
    #plt.show()
    '''
    plt.subplot(221)
    plt.grid(ls = "--", lw = 0.5, color = "#4E616C")
    for idx,strategy in enumerate(STRATEGY): 
        plt.plot(all_var,all_multi_qoe[idx],MARK[idx])
    plt.xlabel(variable)
    plt.ylabel('Average QoE') 
    plt.legend(labels=STRATEGY, bbox_to_anchor=(1,1.2),borderaxespad = 0., ncol = 6)
    
    
    plt.subplot(222)
    plt.grid(ls = "--", lw = 0.5, color = "#4E616C")
    for idx,strategy in enumerate(STRATEGY):
        plt.plot(all_var,all_multi_quality[idx],MARK[idx])
    plt.xlabel(variable)  
    plt.ylabel('Average Bitrate (kbps)')  
    #plt.title('Quality v.s.'+variable)
  
    plt.subplot(223)
    plt.grid(ls = "--", lw = 0.5, color = "#4E616C")
    for idx,strategy in enumerate(STRATEGY):
        plt.plot(all_var,all_multi_miss[idx],MARK[idx])
    plt.xlabel(variable)  
    plt.ylabel('Missdeadline Ratio (%)')  
    #plt.title('Missdeadline_ratio v.s.'+variable)
    
    plt.subplot(224)
    plt.grid(ls = "--", lw = 0.5, color = "#4E616C")
    for idx,strategy in enumerate(STRATEGY):
        if strategy != 'NO_CACHE':
            plt.plot(all_var,all_multi_traffic[idx],MARK[idx])
    plt.xlabel(variable)  
    plt.ylabel("Backhaul Traffic Reduction (GB)") 

    
    plt.subplot(232)
    for idx,strategy in enumerate(STRATEGY):
        plt.plot(all_var,all_multi_jain[idx],MARK[idx])
    plt.legend(STRATEGY)
    plt.xlabel(variable)  
    plt.ylabel("Jain's fairness index") 
    
    plt.subplot(235)
    for idx,strategy in enumerate(STRATEGY):
        plt.plot(all_var,all_multi_hit[idx],MARK[idx])
    plt.legend(STRATEGY)
    plt.xlabel(variable)  
    plt.ylabel('Cachehit_ratio')  

    plt.subplot(236)
    for idx,strategy in enumerate(STRATEGY):
        plt.plot(all_var,all_multi_remain[idx],MARK[idx])
    plt.legend(STRATEGY)
    plt.xlabel(variable)  
    plt.ylabel('Reman_time')  
    #plt.title('Missdeadline_ratio v.s.'+variable)


    #plt.legend(STRATEGY)
    plt.show()
    '''



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
    all_qoe_0 = []
    all_qoe_1 = []
    all_qoe_2 = []
    for strategy in STRATEGY:
        var,multi_qoe,multi_jain,multi_quality,multi_miss,multi_hit,multi_remain,multi_traffic,multi_qoe_0,multi_qoe_1,multi_qoe_2 = result_analyse.result_analyse(strategy,variable)
        all_multi_qoe.append(multi_qoe)
        all_multi_jain.append(multi_jain)
        all_multi_quality.append(multi_quality)
        all_multi_miss.append(multi_miss)
        all_multi_hit.append(multi_hit)
        all_multi_remain.append(multi_remain)
        all_multi_traffic.append(multi_traffic)
        all_qoe_0.append(multi_qoe_0)
        all_qoe_1.append(multi_qoe_1) 
        all_qoe_2.append(multi_qoe_2)       
        
    '''
    our_qoe = np.mean(all_multi_qoe[1])
    performance = []
    for multi_qoe in all_multi_qoe:
        performance.append((our_qoe - np.mean(multi_qoe))/our_qoe)
    '''
    our_qoe = min(all_multi_qoe[1])
    performance = []        
    for multi_qoe in all_multi_qoe:
        performance.append((our_qoe - min(multi_qoe))/our_qoe)    
    #all_var.sort()

    data = {'var':var,'all_multi_qoe':all_multi_qoe,'all_multi_quality':all_multi_quality,'all_multi_miss':all_multi_miss,'all_multi_traffic':all_multi_traffic,'performance':performance,'qoe_0':all_qoe_0,'qoe_1':all_qoe_1,'qoe_2':all_qoe_2,'all_multi_jain':all_multi_jain}
    path = './result/'+str(variable)+'.json'
    with open(path,"w") as file:
        json.dump(data,file)
 
 



def main():
	#all_viewport = load_viewport.load_all_viewport() 
	#all_userview = load_userview.load_all_userview()
	#for variable in VARIABLE:
		#running(variable)
		#analyse(variable)  
		#plot_result(variable) 
	#running('Average Bandwidth (Mbps)')
	#analyse('Average Bandwidth (Mbps)')
	#plot_result('Average Bandwidth (Mbps)') 
	#plot_result('zipf') 
	#plot_bar3()
	plot_cdf('multi_network') 
  #plot_bar('Average Bandwidth (Mbps)')
  #plot_psnr()
	#plot_bar2(1)

if __name__ == '__main__':
	main()