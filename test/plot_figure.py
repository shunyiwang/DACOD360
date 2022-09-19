#coding:utf-8
import os
import json
import math
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import csv
import seaborn as sns
STRATEGY = ['DADS360','DRL_ONLY','GAME_ONLY','TJCD360','NO_CACHE']
def get_cdf_info(strategy,variable,value):
    cdf_qoe = []
    with open('./result/'+str(variable)+'/'+str(value)+'/0/'+str(strategy)+'.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cdf_qoe.append(math.log(float(row['qoe'])))
    return cdf_qoe

def 





def plot_cdf(variable,value):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for strategy in STRATEGY:
        cdf_qoe = get_cdf_info(strategy,variable,value)
        values, base = np.histogram(cdf_qoe,bins=200)
        cumulative = np.cumsum(values)
        ax.plot(base[:-1], cumulative/200.0)
    plt.legend(STRATEGY)
    plt.ylabel('CDF')
    plt.xlabel('QoE')
    plt.grid(ls = "--", lw = 0.5, color = "#4E616C")
    plt.show()
    
def get_boxplot_info(strategy,variable,value,user_idx):
    boxplot_qoe = []
    with open('./csv./response/'+str(variable)+'/'+str(value)+'/'+str(strategy)+'/0/'+str(user_idx)+'.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            #print(row['quality'])        
            boxplot_qoe.append(float(row['quality']))
    return boxplot_qoe
    
def plot_boxplot():
    user_idx = [1,5,10,15]
    plt.subplot(221)
    data = []
    for strategy in STRATEGY:
        boxplot_quality = get_boxplot_info(strategy,'Average Bandwidth (Mbps)',800,user_idx[0])
        data.append(boxplot_quality)		
    plt.boxplot(data,labels=STRATEGY,showmeans=True,showfliers=False) 
    
    plt.subplot(222)
    data = []
    for strategy in STRATEGY:
        boxplot_quality = get_boxplot_info(strategy,'Average Bandwidth (Mbps)',800,user_idx[1])
        data.append(boxplot_quality)		
    plt.boxplot(data,labels=STRATEGY,showmeans=True,showfliers=False)   
    
    
    plt.subplot(223)      
    data = []
    for strategy in STRATEGY:
        boxplot_quality = get_boxplot_info(strategy,'Average Bandwidth (Mbps)',800,user_idx[2])
        data.append(boxplot_quality)		
    plt.boxplot(data,labels=STRATEGY,showmeans=True,showfliers=False) 
    
    plt.subplot(224)
    data = []
    for strategy in STRATEGY:
        boxplot_quality = get_boxplot_info(strategy,'Average Bandwidth (Mbps)',800,user_idx[3])
        data.append(boxplot_quality)		
    plt.boxplot(data,labels=STRATEGY,showmeans=True,showfliers=False)      
  
    
    plt.show()



if __name__ == '__main__':
    #cdf_qoe = get_cdf_info('DADS360','Average Bandwidth (Mbps)',1000)
    #plot_cdf('Average Bandwidth (Mbps)',1000)
    #boxplot_qoe = get_boxplot_info('DADS360','Average Bandwidth (Mbps)',1000,2)
    plot_boxplot()