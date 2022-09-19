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
import multiprocessing as mp     #multiprocessing����Python�еĶ���̹����
os.environ['CUDA_VISIBLE_DEVICES']=''
import tensorflow as tf
import a3c
import pandas as pd
import random


CACHE_MAX = parameter.get_config('CACHE_MAX')
SLEEP_TIME = parameter.get_config('SLEEP_TIME')
VIDEO_NUM = parameter.get_config('VIDEO_NUM')
TILE_SVC_BIT_RATE = parameter.get_config('TILE_SVC_BIT_RATE')
TILE_MAX = parameter.get_config('TILE_MAX')
QUALITY_MAX = parameter.get_config('QUALITY_MAX')


S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 4 #action
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
NUM_AGENTS = 16 #agent��Ŀ
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100   #ÿ100�α���һ��RLģ��
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
TEST_LOG_FOLDER = './test_results/'
TRAIN_TRACES = './cooked_traces/'
#BW_CAPACITY = [100,200,300,400,500,600,700]
BW_CAPACITY = [300,400,500]

# NN_MODEL = './results/pretrain_linear_reward.ckpt'
NN_MODEL = None

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
    
    
def content_delivery(time_stamp,start_mark,info_cluster,users,request_queue,cache_table,user_num,net_env,all_response,average_bandwidth):
    if start_mark == 'first':
        for request in request_queue:
            if request['video_type'] == 'LIVE':
                break
        request['quality'] = 0
        info_cluster = [{'request_cluster':[request]}]  
        request, response, info_cluster = get_response(time_stamp,info_cluster)            
        segment_delay, tans_delay, backhaul_bandwidth, cache_miss, traffic_load_reduction = net_env.download_segment_time(time_stamp,request,response,cache_table)#����segment 
        response = update_response(response,segment_delay,backhaul_bandwidth,cache_miss,traffic_load_reduction)#����response       
        users, request_queue, response = multi_users.update_request_queue(time_stamp,users,request_queue,request,response,cache_table,user_num)
        predict_backhaul_bandwidth = backhaul_bandwidth
        average_bandwidth = backhaul_bandwidth    
        all_response[response['user_idx']].append(response)
        reward = response['qoe']
    else:
        reward = 0
        size = 0
        delay = 0
        while(info_cluster != []):
            #print(info_cluster[0]['quality'])        
            request, response, info_cluster = get_response(time_stamp,info_cluster)            
            segment_delay, tans_delay, backhaul_bandwidth, cache_miss, traffic_load_reduction = net_env.download_segment_time(time_stamp,request,response,cache_table)
            response = update_response(response,segment_delay,backhaul_bandwidth,cache_miss,traffic_load_reduction)#����response       
            time_stamp = time_stamp + tans_delay    #����ʱ���
            users, request_queue, response = multi_users.update_request_queue(time_stamp,users,request_queue,request,response,cache_table,user_num)                    
            #predict_backhaul_bandwidth,average_bandwidth = predict_bandwidth(response,1)    
            all_response[response['user_idx']].append(response)
            reward += response['qoe']
            #print(info_cluster[0]['quality'],request['quality'],response['quality'],response['qoe'])
            size += tans_delay*backhaul_bandwidth
            delay += tans_delay
        reward = reward/user_num                                                    
        predict_backhaul_bandwidth = size/delay                    
        average_bandwidth = ((time_stamp - delay)*average_bandwidth+size)/time_stamp                                   
    return time_stamp, users, request_queue, all_response, predict_backhaul_bandwidth, average_bandwidth, reward    
    

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
    user_num = 0
    all_size = [0,0,0,0]
    for cluster in info_cluster:
        for quality in range(QUALITY_MAX):
            all_size[quality] += cluster['size_info'][quality]
        user_num += cluster['request_num']                
    return user_num, all_size



#����RLģ�͡����ȣ�����rl_test.py���ԣ������Խ��д��'./test_results/'��log�ļ���Ȼ�󣬴˺�����'./test_results/'��log�ļ���reward��������д��log_file
def testing(epoch, nn_model, log_file):

    # ɾ�����Խ���ļ���ÿ�δ���ģ�ͣ����²��Խ��
    os.system('rm -r ' + TEST_LOG_FOLDER)#ɾ��ԭ�ļ�
    os.system('mkdir ' + TEST_LOG_FOLDER)#�������ļ�
    
    # run test script
    os.system('python rl_test.py ' + nn_model)  #����rl_test.py������ģ��

    #���'./test_results/'�еĲ��Խ����д�� './results/log'
    # append test performance to the log
    rewards = []
    test_log_files = os.listdir(TEST_LOG_FOLDER)#���'./test_results/'�����ļ�Ŀ¼

    #ȡ/test_results/'��log��־�����һ�����ݣ���reward.
    for test_log_file in test_log_files:
        reward = []
        with open(TEST_LOG_FOLDER + test_log_file, 'rb') as f:
            for line in f:
                parse = line.split()
                try:
                    reward.append(float(parse[-1]))#ȡparse���һ��Ԫ��
                except IndexError:
                    break
        rewards.append(np.sum(reward[1:]))

    rewards = np.array(rewards)

    rewards_min = np.min(rewards)#rewards����Сֵ
    rewards_5per = np.percentile(rewards, 5)#5%�ķ�λ��������rewards������֮���5%λ��
    rewards_mean = np.mean(rewards)#rewards��ƽ��ֵ
    rewards_median = np.percentile(rewards, 50)#50%�ķ�λ��������rewards������֮�����λ��
    rewards_95per = np.percentile(rewards, 95)#95%�ķ�λ��������rewards������֮���95%λ��
    rewards_max = np.max(rewards)#rewards�����ֵ
    '''
    log_file.write(str(epoch) + '\t' +
                   str(rewards_min) + '\t' +
                   str(rewards_5per) + '\t' +
                   str(rewards_mean) + '\t' +
                   str(rewards_median) + '\t' +
                   str(rewards_95per) + '\t' +
                   str(rewards_max) + '\n')
    log_file.flush()
    '''
    log_file.write(str(epoch) + '\t' +
                   str(rewards) + '\n')
    log_file.flush()


#���ȸ�agent��Ϣ�����ϴ���
def central_agent(net_params_queues, exp_queues):

    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    #��ʼ��log_central��־
    logging.basicConfig(filename=LOG_FILE + '_central',
                        filemode='w',
                        level=logging.INFO)

    with tf.Session() as sess, open(LOG_FILE + '_test', 'wb') as test_log_file:#��log_testΪtest_log_file,�ֽ�д��
        #AC����ѵ��
        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        summary_ops, summary_vars = a3c.build_summaries() #������Ϣ�ܽ�

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor.ָ��һ���ļ���������ͼ
        saver = tf.train.Saver()  # save neural net parameters

        #��ʼ������ģ��
        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")

        epoch = 0  #ѵ������

        #16��agentͬ��ѵ���������ݶ�
        # assemble experiences from agents, compute the gradients
        while True:
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            critic_net_params = critic.get_network_params()

            #��ʼ��net_params_queues[i]��iȡ1��16
            for i in xrange(NUM_AGENTS):
                net_params_queues[i].put([actor_net_params, critic_net_params])
                # Note: this is synchronous version of the parallel training,
                # which is easier to understand and probe. The framework can be
                # fairly easily modified to support asynchronous training.
                # Some practices of asynchronous training (lock-free SGD at
                # its core) are nicely explained in the following two papers:
                # https://arxiv.org/abs/1602.01783
                # https://arxiv.org/abs/1106.5730

            # record average_reward and td_loss change
            # in the experiences from the agents
            total_batch_len = 0.0
            total_reward = 0.0
            total_td_loss = 0.0
            total_entropy = 0.0   #��(Entropy)��һ�ִӶ����ĽǶ���������Ϣ(Information) ���ٵ�ָ��,һ����Ϣ���������¼��Ĳ�ȷ����Խ������������Ϣ��Խ�ࡣ
            total_agents = 0.0 

            # assemble experiences from the agents        
            actor_gradient_batch = []
            critic_gradient_batch = []

            #��16��agent��r_batch��td_batch��entropy�ֱ���ӣ�����total_reward��total_td_loss��total_entropy
            for i in xrange(NUM_AGENTS):
                s_batch, a_batch, r_batch, terminal, info = exp_queues[i].get()#.get()��ȡÿ��agent���̵�ֵ

                #����a3c.compute_gradients,����actor_gradient, critic_gradient, td_batch������ actor_gradient_batch = []��critic_gradient_batch = []
                actor_gradient, critic_gradient, td_batch = \
                    a3c.compute_gradients(
                        s_batch=np.stack(s_batch, axis=0),
                        a_batch=np.vstack(a_batch),
                        r_batch=np.vstack(r_batch),
                        terminal=terminal, actor=actor, critic=critic)

                actor_gradient_batch.append(actor_gradient)
                critic_gradient_batch.append(critic_gradient)

                total_reward += np.sum(r_batch)
                total_td_loss += np.sum(td_batch)
                total_batch_len += len(r_batch)
                total_agents += 1.0    #agent+1
                total_entropy += np.sum(info['entropy'])


            # compute aggregated gradient
            assert NUM_AGENTS == len(actor_gradient_batch)
            assert len(actor_gradient_batch) == len(critic_gradient_batch)
            # assembled_actor_gradient = actor_gradient_batch[0]
            # assembled_critic_gradient = critic_gradient_batch[0]
            # for i in xrange(len(actor_gradient_batch) - 1):
            #     for j in xrange(len(assembled_actor_gradient)):
            #             assembled_actor_gradient[j] += actor_gradient_batch[i][j]
            #             assembled_critic_gradient[j] += critic_gradient_batch[i][j]
            # actor.apply_gradients(assembled_actor_gradient)
            # critic.apply_gradients(assembled_critic_gradient)
            
            for i in xrange(len(actor_gradient_batch)):
                actor.apply_gradients(actor_gradient_batch[i])      #minimize()����compute_gradients()��apply_gradients()�����������ļ����
                critic.apply_gradients(critic_gradient_batch[i])  #�ú����������ǽ�compute_gradients()���ص�ֵ��Ϊ���������variable���и���

            # ����ѵ����Ϣ��TD_loss��Avg_reward��Avg_entropy����д��log_central
            epoch += 1
            avg_reward = total_reward  / total_agents
            avg_td_loss = total_td_loss / total_batch_len
            avg_entropy = total_entropy / total_batch_len

            logging.info('Epoch: ' + str(epoch) +                 #д��log_central
                         ' TD_loss: ' + str(avg_td_loss) +
                         ' Avg_reward: ' + str(avg_reward) +
                         ' Avg_entropy: ' + str(avg_entropy))
            print(epoch)
            summary_str = sess.run(summary_ops, feed_dict={      #a3c����summary_vars = [td_loss, eps_total_reward, avg_entropy]
                summary_vars[0]: avg_td_loss,
                summary_vars[1]: avg_reward,
                summary_vars[2]: avg_entropy
            })

            writer.add_summary(summary_str, epoch)    #����add_summary����������ѵ���������ݱ�����filewriterָ�����ļ�SUMMARY_DIR = './results'
            writer.flush()

            #ÿ��100�Σ���ģ��ѵ������д��# NN_MODEL = './results/pretrain_linear_reward.ckpt'
            if epoch % MODEL_SAVE_INTERVAL == 0:
                # Save the neural net parameters to disk.
                save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" +
                                       str(epoch) + ".ckpt")
                logging.info("Model saved in file: " + save_path)
                testing(epoch,     #����ģ�ͣ����д��log_test
                    SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt", 
                    test_log_file)

#ÿ��agentѵ��������env���������ݣ���central_agent�б�����reward��
def agent(agent_id, net_params_queue, exp_queue):

    with tf.Session() as sess, open(LOG_FILE + '_agent_' + str(agent_id), 'wb') as log_file:          #log_fileΪlog_agent_0...
        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        #��ʼ���ӽ���
        # initial synchronization of the network parameters from the coordinator
        actor_net_params, critic_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)
        critic.set_network_params(critic_net_params)
        np.random.seed(RANDOM_SEED)

        s_batch = []
        a_batch = []
        r_batch = []  
        entropy_record = []


        bw_fluctuation = 40
        user_num = 200
        cache_max = 13
        trace_idx = 0        
        bw_capacity = BW_CAPACITY[trace_idx]
        
        
        #�������ԣ�ֱ��������������Ƶ
        all_viewport = load_viewport.load_all_viewport() 
        all_userview = load_userview.load_all_userview()
        time_stamp = 0
        net_env = init_network('real',bw_capacity/1000.0,bw_fluctuation/10)
        users = multi_users.init_multi_users(user_num)
        cache_table = cache_update.init_cache(cache_max*CACHE_MAX/100.0)
        request_queue = multi_users.init_request_queue(users,user_num,cache_table)
        info_cluster = []
        last_action = 0    
        average_bandwidth = 0
        all_response = [[] for j in range(user_num)]


        #��ʼ����һ������
        time_stamp, users, request_queue, all_response, predict_backhaul_bandwidth, average_bandwidth, _ = content_delivery(time_stamp,'first',info_cluster,users,request_queue,cache_table,user_num,net_env,all_response,average_bandwidth)              
        
        #������Ƶ��ѵ��agent,ֱ����Ƶ����  
        while True:  # experience video streaming forever
            if len(s_batch) == 0:
                state = np.zeros((S_INFO, S_LEN))
            else:
                state = np.array(s_batch[-1], copy=True)
            # dequeue history record
            state = np.roll(state, -1, axis=1)
            
            info_cluster = queue_aggregation.cluster(request_queue,predict_backhaul_bandwidth,cache_table)
            user_num, all_size = get_info(info_cluster)
            min_remain_time = info_cluster[0]['remain_time']
            ALL_SIZE = (float(user_num*max(TILE_SVC_BIT_RATE))*float(TILE_MAX)/8.0)  

            # this should be S_INFO number of terms
            state[0, -1] = TILE_SVC_BIT_RATE[last_action] / float(np.max(TILE_SVC_BIT_RATE))  # last quality
            state[1, -1] = min_remain_time / 20.0  # 10 sec
            state[2, -1] = float(predict_backhaul_bandwidth) / predict_backhaul_bandwidth  # kilo byte / ms
            #state[3, :A_DIM] = np.array(qoe_info) / float(np.max(BITRATE))  # 10 sec
            state[4, :A_DIM] = np.array(all_size) / float(ALL_SIZE)  # mega byte
            #state[5, -1] = (cache[video_idx]) / 3.0  

            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
            action_cumsum = np.cumsum(action_prob)
            action = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()

            entropy_record.append(a3c.compute_entropy(action_prob[0]))            
            presupposed_bw = all_size[action]
            #print(presupposed_bw)                
            info_cluster = gamebased_allocating.gamebased_allocating(info_cluster,presupposed_bw)             
            time_stamp, users, request_queue, all_response, predict_backhaul_bandwidth, average_bandwidth, reward = content_delivery(time_stamp,'other',info_cluster,users,request_queue,cache_table,user_num,net_env,all_response,average_bandwidth)                   

            print(action_prob)
 
            end_of_video = (not if_continue_playback(users))

            r_batch.append(reward)
                                                                   
            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write(str(time_stamp) + '\t' +
                           str(TILE_SVC_BIT_RATE[last_action]) + '\t' +
                           str(min_remain_time) + '\t' +
                           str(predict_backhaul_bandwidth) + '\t' + 
                           #str(np.array(qoe_info)) + '\t' +  
                           #str(np.array(delay_info)) + '\t' + 
                           #str(cache[video_idx]) + '\t' + 
                           str(np.array(all_size)) + '\t' +
                           str(action) + '\t' +
                           str(reward) + '\n')
            log_file.flush()

            last_action = action    
            s_batch.append(state)
            action_vec = np.zeros(A_DIM)
            action_vec[action] = 1
            a_batch.append(action_vec)
            
            if end_of_video:
                exp_queue.put([s_batch[:],  # ignore the first chuck
                               a_batch[:],  # since we don't have the
                               r_batch[:],  # control over it
                               end_of_video,
                               {'entropy': entropy_record}])

                # synchronize the network parameters from the coordinator
                actor_net_params, critic_net_params = net_params_queue.get()
                actor.set_network_params(actor_net_params)
                critic.set_network_params(critic_net_params)

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]
                del entropy_record[:]

                log_file.write('\n')  # so that in the log we know where video ends    
                                        

            
            # store the state and action into batches
            if end_of_video:
                last_action = 0               
                time_stamp = 0
                trace_idx = (trace_idx+1) % len(BW_CAPACITY)        
                bw_capacity = BW_CAPACITY[trace_idx]
                net_env = init_network('real',bw_capacity/1000.0,bw_fluctuation/10)                    
                users = multi_users.init_multi_users(user_num)
                request_queue = multi_users.init_request_queue(users,user_num,cache_table)
                info_cluster = []
                average_bandwidth = 0    
                presupposed_bw = 0  
                all_response = [[] for j in range(user_num)]
                time_stamp, users, request_queue, all_response, predict_backhaul_bandwidth, average_bandwidth, _ = content_delivery(time_stamp,'first',info_cluster,users,request_queue,cache_table,user_num,net_env,all_response,average_bandwidth)



def main():

    np.random.seed(RANDOM_SEED)

    # create result directory
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in xrange(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))   #��ʼ��16���ӽ��̶��У�mp.Queue��ʼ��һ��Queue�������ɽ���1��put��Ϣ
        exp_queues.append(mp.Queue(1))

    #tf.train.Coordinator() ������һ���߳�Э��������������֮����Session�������������߳�
    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,      #��������central_agent
                             args=(net_params_queues, exp_queues))
    coordinator.start()

    #��ʼ������agent
    agents = []
    for i in xrange(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i,
                                       net_params_queues[i],
                                       exp_queues[i])))
    for i in xrange(NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()


if __name__ == '__main__':
    main()
