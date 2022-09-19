# -*-coding:utf-8 -*-
import random
import math
import matplotlib.pyplot as plt
import config as parameter
import numpy as np
import time
import config as parameter
period = 20 #��������
#population_size = 10 #��Ⱥ��ģ
pc = 0.6
pm = 0.1
feasible_max = 5
gama = 0.9
TILE_BIT_RATE = parameter.get_config('TILE_BIT_RATE')


def species_origin(choromosome_length, new_request_queue):#��ʼ����Ⱥ ����chromosome_length��С��population_size���������Ⱥ
	population=[]
	population_size = choromosome_length
	#population_size = 5
	population.append(encoding(new_request_queue))
	for i in range(population_size-1):
		utility = 1.0
		feasible = 0
		while utility <= 1.0:
			chromosome2=[]#Ⱦɫ���ݴ���
			for j in range(choromosome_length):
				chromosome2.append(random.randint(0,1))#�������һ��Ⱦɫ��,�ɶ����������
			chromosome = decoding(chromosome2)
			utility = predict_utility(chromosome, new_request_queue)
			feasible += 1
			if feasible > feasible_max:
				chromosome = np.zeros(choromosome_length).tolist()
				break
		population.append(chromosome2)#��Ⱦɫ����ӵ���Ⱥ��
	return population# ����Ⱥ���أ���Ⱥ�Ǹ���ά���飬�����Ⱦɫ����ά

def decoding(chromosome2):
	chromosome = []
	for j in range(len(chromosome2)): #�ӵ�һ������ʼ���������Ʊ���ת������������ ��1110->32
		if j % 2 == 0:
			quality = chromosome2[j]*2
		else:
			quality += chromosome2[j]
			chromosome.append(quality)
	return chromosome


def translation(population,choromosome_length): #����  input:��Ⱥ,Ⱦɫ�峤�� ������̾��ǽ���Ԫ����ת����һԪ�����Ĺ���
	assert choromosome_length % 2 == 0
	all_chromosome=[]
	for i in range(len(population)):
		chromosome2=[]
		for j in range(choromosome_length): #�ӵ�һ������ʼ���������Ʊ���ת������������ ��1110->32
			if j % 2 == 0:
				quality = population[i][j]*2
			else:
				quality += population[i][j]
				chromosome2.append(quality)
		all_chromosome.append(chromosome2) #һ��Ⱦɫ�������ɣ���һ��������������Ϊ��������
	return all_chromosome # ������Ⱥ�����и���


def encoding(new_request_queue):
	chromosome2 = []
	#print(request_queue_cluster) 
	for cluster in new_request_queue:
		quality = cluster['quality']
		chromosome2.append(quality//2)
		chromosome2.append(quality%2)
	return chromosome2

def calculate_fitness(population,choromosome_length,new_request_queue):# Ŀ�꺯���൱�ڻ��� ��Ⱦɫ�����ɸѡ
	all_fitness = []
	all_chromosome = translation(population,choromosome_length)
	for chromosome in all_chromosome:
		utility = predict_utility(chromosome,new_request_queue)
		all_fitness.append(utility)
	return all_fitness

def predict_utility(chromosome, new_request_queue):#����ÿ��Ⱦɫ���Ч�ã���qoe�����˻���������deadline��Ϊ0
	utility = 1.0
	size = 0.0
	predict_bandwidth = new_request_queue[0]['predict_bandwidth']
	for idx,quality in enumerate(chromosome):
		if new_request_queue[idx]['size_info'][quality] + size > predict_bandwidth:
			utility = 1.0
			break
		else:
			utility += new_request_queue[idx]['qoe_info'][quality]
			size += new_request_queue[idx]['size_info'][quality]
	return utility


def cumsum(all_fitness):#������Ӧ��쳲������б�
	for i in range(len(all_fitness)-2,-1,-1):# range(start,stop,[step])
		total = 0.0
		j = 0# ������
		while(j<=i):
			total += all_fitness[j]
			j += 1
		all_fitness[i] = total
		all_fitness[len(all_fitness)-1] = 1
	return all_fitness

def selection(population,all_fitness,choromosome_length):#3.ѡ����Ⱥ�и�����Ӧ�����ĸ���
	population_size = choromosome_length
	zip_pop = zip(population,all_fitness)
	sort_zipped = sorted(zip_pop,key=lambda x:(x[1]),reverse = True)
	sort_zipped = sort_zipped[:population_size]
	result = zip(*sort_zipped)
	population, all_fitness = [list(x) for x in result]
	#print(population, all_fitness)
	return population,all_fitness


def crossover(population, new_request_queue):#�������,pc�Ǹ�����ֵ,���õ��㽻��
	pop_len = len(population)
	for i in range(pop_len-1):
		if random.random() < pc:
			utility = 1.0
			feasible = 0
			while utility <= 1.0 and feasible < feasible_max:
				cpoint = random.randrange(0,len(population[0]),2)#����Ⱥ������������ɵ��㽻��� ��Ϊż��λ�� 
				temporary1 = []
				temporary2 = []
				temporary1.extend(population[i][0:cpoint])
				#print(population[i][0:cpoint])
				temporary1.extend(population[i+1][cpoint:len(population[i])])
				#��tmporary1��Ϊ�ݴ�������ʱ��ŵ�i��Ⱦɫ���е�ǰ0��cpoint������
				#Ȼ���ٰѵ�i+1��Ⱦɫ���еĺ�cpoint����i��Ⱦɫ���еĻ�����������䵽temporary2����
				temporary2.extend(population[i+1][0:cpoint])
				temporary2.extend(population[i][cpoint:len(population[i])])
				# ��tmporary2��Ϊ�ݴ�������ʱ��ŵ�i+1��Ⱦɫ���е�ǰ0��cpoint������
				# Ȼ���ٰѵ�i��Ⱦɫ���еĺ�cpoint����i��Ⱦɫ���еĻ�����������䵽temporary2����
				chromosome2_temporary1 = decoding(temporary1)
				chromosome2_temporary2 = decoding(temporary2)
				utility1 = predict_utility(chromosome2_temporary1, new_request_queue)
				utility2 = predict_utility(chromosome2_temporary2, new_request_queue)
				utility = min(utility1,utility2)
				feasible += 1
			if utility > 1.0:
				population.append(temporary1)
				population.append(temporary2)
			# ��i��Ⱦɫ��͵�i+1��Ⱦɫ���������/�������
	return population

def mutation(population, new_request_queue):     # pm�Ǹ�����ֵ
	px = len(population)    # �����Ⱥ��������Ⱥ/����ĸ���
	py = len(population[0])    # Ⱦɫ��/�������ĸ���
	for i in range(px):
		if random.random() < pm:
			utility = 1.0
			feasible = 0
			while utility <= 1.0 and feasible < feasible_max:
				mpoint = random.randint(0,py-1)
				temporary = population[i][:]
				if temporary[mpoint] == 1:#��mpoint��������е���������죬��Ϊ0����1
					temporary[mpoint] = 0
				else:
					temporary[mpoint] = 1
				chromosome = decoding(temporary)
				utility = predict_utility(chromosome, new_request_queue)
				feasible += 1
			if utility > 1.0:
				population.append(temporary)
	return population

def best(population,fitness_value):#Ѱ����õ���Ӧ�Ⱥ͸���
	px = len(population)
	bestindividual = population[0]
	bestfitness = fitness_value[0]
	for i in range(1,px):# ѭ���ҳ�������Ӧ�ȣ���Ӧ������Ҳ������õĸ���
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

def bitrate_allocating(new_request_queue):
	best_utility = 1.0
	choromosome_length = len(new_request_queue) * 2
	results = []
	population = pop = species_origin(choromosome_length, new_request_queue)
	for i in range(period):
		crossover(population, new_request_queue)
		mutation(population, new_request_queue)
		all_fitness = calculate_fitness(population,choromosome_length,new_request_queue)
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
	for j in range(len(best_result[1])): #ת��������
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
			new_request_queue[idx]['quality'] = quality
	if utility == 1:
		for idx,quality in enumerate(chromosome):
			new_request_queue[idx]['quality'] = 0                   
			#print(one_request_queue)    
	#print(cluster_request_queue)
	#print(chromosome,best_utility)
	return new_request_queue

if __name__ == '__main__':
	request_queue = []
	for i in range(3):
		request = {'remain_time':4-i,'predict_segment_delay':i,'quality':0}
		request_queue.append(request)
	request_queue,utility = find_bestquality(request_queue)
	print(request_queue)