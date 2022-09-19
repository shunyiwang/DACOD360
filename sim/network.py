#coding:utf-8
import math
import numpy as np
from load_network import load_network
import multi_users as multi_users
import config as parameter

MILLISECONDS_IN_SECOND = 1000.0
KB_IN_GB = 1000000.0
BITS_IN_BYTE = 8.0
RTT_USER = 0.0
RTT_backhaul = 0.0
RTT_fronthaul = 0.0
RANDOM_SEED = parameter.get_config('RANDOM_SEED')
PACKET_PAYLOAD_PORTION = parameter.get_config('PACKET_PAYLOAD_PORTION')
POSITIONDICT = parameter.get_config('POSITIONDICT')
TILE_BIT_RATE = parameter.get_config('TILE_BIT_RATE')  # Kbps AVC编码
video_names = parameter.get_config('video_names')
RRH_MAX = parameter.get_config('RRH_MAX')




class Environment:
    def __init__(self, bw_capacity, all_time, all_cooked_bw):
        #assert len(all_time) == len(all_backhaul_bw)
        #assert len(all_time) == len(all_fronthaul_bw)

        self.all_time = all_time
        self.all_cooked_bw = all_cooked_bw

        # pick a random trace file
        self.trace_idx = 0
        self.cooked_time = self.all_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]
        
        self.backhaul_bw = []
        for bw in self.cooked_bw:
            self.backhaul_bw.append(bw*bw_capacity)
            #self.backhaul_bw.append(simulated_bw)
        #print(self.backhaul_bw)         
                 
        self.mahimahi_start_ptr = 1
        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = 1
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.virtual_mahimahi_ptr = self.mahimahi_ptr
        self.virtual_last_mahimahi_time = self.last_mahimahi_time

    def reset_download_time(self):
        self.virtual_mahimahi_ptr = self.mahimahi_ptr
        self.virtual_last_mahimahi_time = self.last_mahimahi_time


    def download_tile_time(self, time_stamp, tile_size):
        tile_delay = 0.0  # in s
        #print(self.all_time)        
        self.virtual_last_mahimahi_time = time_stamp                
        self.virtual_mahimahi_ptr = int(math.floor(time_stamp))                      
        video_tile_counter_sent = 0.0  # in bytes
        while True:  # download video tile over mahimahi
            throughput = self.backhaul_bw[self.virtual_mahimahi_ptr] \
                             * KB_IN_GB / BITS_IN_BYTE
            duration = self.virtual_mahimahi_ptr + 1 \
                       - self.virtual_last_mahimahi_time
            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION
            if video_tile_counter_sent + packet_payload > tile_size:
                fractional_time = (tile_size - video_tile_counter_sent) / \
                                  throughput / PACKET_PAYLOAD_PORTION
                tile_delay += fractional_time
                self.virtual_last_mahimahi_time += fractional_time
                break
            video_tile_counter_sent += packet_payload
            tile_delay += duration
            self.virtual_mahimahi_ptr += 1
            self.virtual_last_mahimahi_time = self.virtual_mahimahi_ptr

            #if self.virtual_mahimahi_ptr >= len(self.all_time):
                # loop back in the beginning
                # note: trace file starts with time 0
                #self.virtual_mahimahi_ptr = 1
                #self.virtual_last_mahimahi_time = 0
            #tile_delay *= MILLISECONDS_IN_SECOND      
        return tile_delay

    def download_segment_time(self,time_stamp,request,response,cache_table):
        viewport = request['viewport']
        video_idx = video_names.index(request['video'])
        #print(request['video'],response['quality'])
        rrh = request['associative_rrh']
        segment_idx = request['segment_idx']     
        tile_size = request['tile_size']  
        cluster_mark = request['cluster_mark']
        video_type = request['video_type']       
        backhaul_dealy = 0
        cache_miss = 0
        backhaul_bandwidth = 0
        trans_dealy = 0
        traffic_load_reduction = 0
        backhaul_tile_size = 0
        if cluster_mark == 0:
            for tile_idx in viewport:
                if cache_table[video_idx][segment_idx][tile_idx] < response['quality']:
                    for quality in range(max(cache_table[video_idx][segment_idx][tile_idx]+1,0),response['quality']+1):
                        backhaul_tile_size += tile_size[quality]
                    for quality in range(0,max(cache_table[video_idx][segment_idx][tile_idx]+1,0)):
                        traffic_load_reduction += tile_size[quality]
                    cache_miss += 1
                else:
                    for quality in range(0,response['quality']+1):
                        traffic_load_reduction += tile_size[quality]                                      
            trans_dealy = self.download_tile_time(time_stamp,backhaul_tile_size)
        elif cluster_mark == 1:
            trans_dealy = 0
            traffic_load_reduction = len(viewport)*tile_size[response['quality']]
        elif cluster_mark == 2:
            for tile_idx in viewport: 
                if video_type == 'VOD':
                    for quality in range(1,response['quality']+1):
                        backhaul_tile_size += tile_size[quality]
                else:
                    for quality in range(0,response['quality']+1):
                        backhaul_tile_size += tile_size[quality]                                                                                                        
            trans_dealy = self.download_tile_time(time_stamp,backhaul_tile_size)          
        if trans_dealy == 0:
            segment_delay = RTT_fronthaul
        else:
            segment_delay = trans_dealy + RTT_backhaul
            backhaul_bandwidth = backhaul_tile_size / trans_dealy
        return segment_delay, trans_dealy, backhaul_bandwidth, cache_miss, traffic_load_reduction

