# -*-coding:utf-8 -*-
import random
import math
import matplotlib.pyplot as plt
import config as parameter
import numpy as np
import time
import config as parameter
period = 20 #迭代次数
#population_size = 10 #种群规模
pc = 0.6
pm = 0.1
feasible_max = 5
gama = 0.9
TILE_BIT_RATE = parameter.get_config('TILE_BIT_RATE')


def species_origin(choromosome_length, cluster_request_queue):#初始化种群 生成chromosome_length大小的population_size个个体的种群
	population=[]
	population_size = choromosome_length
	#population_size = 5
	population.append(encoding(cluster_request_queue))
	for i in range(population_size-1):
		utility = 1.0
		feasible = 0
		while utility <= 1.0:
			chromosome2=[]#染色体暂存器
			for j in range(choromosome_length):
				chromosome2.append(random.randint(0,1))#随机产生一个染色体,由二进制数组成
			chromosome = decoding(chromosome2)
			utility = predict_utility(chromosome, cluster_request_queue)
			feasible += 1
			if feasible > feasible_max:
				chromosome = np.zeros(choromosome_length).tolist()
				break
		population.append(chromosome2)#将染色体添加到种群中
	return population# 将种群返回，种群是个二维数组，个体和染色体两维

def decoding(chromosome2):
	chromosome = []
	for j in range(len(chromosome2)): #从第一个基因开始，将二进制编码转化成码率序列 如1110->32
		if j % 2 == 0:
			quality = chromosome2[j]*2
		else:
			quality += chromosome2[j]
			chromosome.append(quality)
	return chromosome


def translation(population,choromosome_length): #编码  input:种群,染色体长度 编码过程就是将多元函数转化成一元函数的过程
	assert choromosome_length % 2 == 0
	all_chromosome=[]
	for i in range(len(population)):
		chromosome2=[]
		for j in range(choromosome_length): #从第一个基因开始，将二进制编码转化成码率序列 如1110->32
			if j % 2 == 0:
				quality = population[i][j]*2
			else:
				quality += population[i][j]
				chromosome2.append(quality)
		all_chromosome.append(chromosome2) #一个染色体编码完成，由一个二进制数编码为码率序列
	return all_chromosome # 返回种群中所有个体


def encoding(request_queue_cluster):
	chromosome2 = []
	#print(request_queue_cluster) 
	for one_request_queue in request_queue_cluster:
		quality = one_request_queue[0]['quality']
		chromosome2.append(quality//2)
		chromosome2.append(quality%2)
	return chromosome2

def calculate_fitness(population,choromosome_length,cluster_request_queue):# 目标函数相当于环境 对染色体进行筛选
	all_fitness = []
	all_chromosome = translation(population,choromosome_length)
	for chromosome in all_chromosome:
		utility = predict_utility(chromosome,cluster_request_queue)
		all_fitness.append(utility)
	return all_fitness

def predict_utility(chromosome, cluster_request_queue):#计算每条染色体的效用，即qoe的连乘积，如果错过deadline则为0
	utility = 1.0
	used_time = 0.0
	for idx,quality in enumerate(chromosome):
		one_request_queue = cluster_request_queue[idx]
		utility_list = []
		one_utility = 0
		for request in one_request_queue:
			segment_delay = request['delay_info'][quality]
			if request['remain_time'] - segment_delay - used_time < 0.0: #超过deadline qoe为1
				one_utility = 1.0    
				break
			else:
				utility_list.append(request['qoe_info'][quality])
				one_utility = request['qoe_info'][quality]
				#utility_list.append(TILE_BIT_RATE[quality])
				#print(request['average_popularity'],request['qoe_info'][quality])        
				used_time += request['delay_info'][quality]    
		if one_utility == 1.0:
			utility = 1.0
			break
		else:
			utility += one_utility
	#print(utility)
	return utility


def cumsum(all_fitness):#计算适应度斐伯纳且列表
	for i in range(len(all_fitness)-2,-1,-1):# range(start,stop,[step])
		total = 0.0
		j = 0# 倒计数
		while(j<=i):
			total += all_fitness[j]
			j += 1
		all_fitness[i] = total
		all_fitness[len(all_fitness)-1] = 1
	return all_fitness
'''
def selection(population,all_fitness):#3.选择种群中个体适应度最大的个体
	new_fitness = []    #单个公式暂存器
	total_fitness = sum(all_fitness)    #将所有的适应度求和
	for i in range(len(all_fitness)):
		new_fitness.append(all_fitness[i] / total_fitness)    #将所有个体的适应度正则化
	new_fitness = cumsum(new_fitness)
	print(new_fitness)
	ms = []    #存活的种群
	#population_length = pop_len = len(population)    #求出种群长度 根据随机数确定哪几个能存活
	for i in range(population_size):    # 产生种群个数的随机值
		ms.append(random.random())
	ms.sort()    # 存活的种群排序
	fitin = 0
	newin = 0
	new_pop = []
	new_fit = []
	while newin < population_size:    #轮盘赌方式
		if(ms[newin] <= new_fitness[fitin]):
			new_pop.append(population[fitin])
			new_fit.append(all_fitness[fitin])
			newin += 1
		else:
			fitin += 1
	population = new_pop
	all_fitness = new_fit
	#print(population)
	return population,all_fitness
'''
def selection(population,all_fitness,choromosome_length):#3.选择种群中个体适应度最大的个体
	population_size = choromosome_length
	zip_pop = zip(population,all_fitness)
	sort_zipped = sorted(zip_pop,key=lambda x:(x[1]),reverse = True)
	sort_zipped = sort_zipped[:population_size]
	result = zip(*sort_zipped)
	population, all_fitness = [list(x) for x in result]
	#print(population, all_fitness)
	return population,all_fitness


def crossover(population, cluster_request_queue):#交叉操作,pc是概率阈值,采用单点交叉
	pop_len = len(population)
	for i in range(pop_len-1):
		if random.random() < pc:
			utility = 1.0
			feasible = 0
			while utility <= 1.0 and feasible < feasible_max:
				cpoint = random.randrange(0,len(population[0]),2)#在种群个数内随机生成单点交叉点 仅为偶数位点 
				temporary1 = []
				temporary2 = []
				temporary1.extend(population[i][0:cpoint])
				#print(population[i][0:cpoint])
				temporary1.extend(population[i+1][cpoint:len(population[i])])
				#将tmporary1作为暂存器，暂时存放第i个染色体中的前0到cpoint个基因，
				#然后再把第i+1个染色体中的后cpoint到第i个染色体中的基因个数，补充到temporary2后面
				temporary2.extend(population[i+1][0:cpoint])
				temporary2.extend(population[i][cpoint:len(population[i])])
				# 将tmporary2作为暂存器，暂时存放第i+1个染色体中的前0到cpoint个基因，
				# 然后再把第i个染色体中的后cpoint到第i个染色体中的基因个数，补充到temporary2后面
				chromosome2_temporary1 = decoding(temporary1)
				chromosome2_temporary2 = decoding(temporary2)
				utility1 = predict_utility(chromosome2_temporary1, cluster_request_queue)
				utility2 = predict_utility(chromosome2_temporary2, cluster_request_queue)
				utility = min(utility1,utility2)
				feasible += 1
			if utility > 1.0:
				population.append(temporary1)
				population.append(temporary2)
			# 第i个染色体和第i+1个染色体基因重组/交叉完成
	return population

def mutation(population, cluster_request_queue):     # pm是概率阈值
	px = len(population)    # 求出种群中所有种群/个体的个数
	py = len(population[0])    # 染色体/个体基因的个数
	for i in range(px):
		if random.random() < pm:
			utility = 1.0
			feasible = 0
			while utility <= 1.0 and feasible < feasible_max:
				mpoint = random.randint(0,py-1)
				temporary = population[i][:]
				if temporary[mpoint] == 1:#将mpoint个基因进行单点随机变异，变为0或者1
					temporary[mpoint] = 0
				else:
					temporary[mpoint] = 1
				chromosome = decoding(temporary)
				utility = predict_utility(chromosome, cluster_request_queue)
				feasible += 1
			if utility > 1.0:
				population.append(temporary)
	return population

def best(population,fitness_value):#寻找最好的适应度和个体
	px = len(population)
	bestindividual = population[0]
	bestfitness = fitness_value[0]
	for i in range(1,px):# 循环找出最大的适应度，适应度最大的也就是最好的个体
		if fitness_value[i] > bestfitness:
			bestfitness = fitness_value[i]
			bestindividual = population[i]
	return bestindividual,bestfitness

def plot(results):
	X = []
	Y = []
	for i in range(period):
		X.append(i)
		Y.append(results[i][0])
	plt.plot(X, Y)
	plt.show()

def find_bestquality(cluster_request_queue,best_utility):
	choromosome_length = len(cluster_request_queue) * 2
	results = []
	population = pop = species_origin(choromosome_length, cluster_request_queue)
	for i in range(period):
		crossover(population, cluster_request_queue)
		mutation(population, cluster_request_queue)
		all_fitness = calculate_fitness(population,choromosome_length,cluster_request_queue)
		#print(len(population))
		#print(all_fitness)
		population,all_fitness = selection(population,all_fitness,choromosome_length)
		#print(all_fitness)
		#print(len(population))
		best_individual, best_fitness = best(population, all_fitness)
		results.append([best_fitness, best_individual])

	#results.sort()
	#print(results)
	#plot(results)
	best_result = max(results)
	#print(results,best_result)
	utility = best_result[0]

	chromosome = []
	for j in range(len(best_result[1])): #转化成码率
		if j % 2 == 0:
			quality = best_result[1][j]*2
		else:
			quality += best_result[1][j]
			chromosome.append(quality)
	#print(best_result)
	#print(chromosome,utility)
	if utility > best_utility:
		best_utility = utility
		for idx,quality in enumerate(chromosome):
			one_request_queue = cluster_request_queue[idx]
			for j,request in enumerate(one_request_queue):
				cluster_request_queue[idx][j]['quality'] = quality
				cluster_request_queue[idx][j]['predict_segment_delay'] = request['delay_info'][quality]  
        
	if utility <= 1:
		for idx,quality in enumerate(chromosome):
				cluster_request_queue[idx][0]['quality'] = 0            
			#print(one_request_queue)    
	#print(cluster_request_queue)
	#print(chromosome,best_utility)
	return cluster_request_queue, best_utility

if __name__ == '__main__':
	request_queue = []
	for i in range(3):
		request = {'remain_time':4-i,'predict_segment_delay':i,'quality':0}
		request_queue.append(request)
	request_queue,utility = find_bestquality(request_queue)
	print(request_queue)