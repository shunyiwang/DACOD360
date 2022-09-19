#coding:utf-8
import os
import json
import math
COOKED_USERVIEW_FOLDER = './user_view/view_video.json'
import config as parameter
VIEW_VIDEO_NUM = parameter.get_config('VIEW_VIDEO_NUM')

def load_all_userview():
    global all_userview
    all_userview = []
    with open(COOKED_USERVIEW_FOLDER,'r') as fp:
        #json_data = fp.read()
        all_userview = json.load(fp)    
    return all_userview

def load_userview(user_idx,video_idx):
    global all_userview
    if video_idx < VIEW_VIDEO_NUM:
        view_video = all_userview[user_idx]['view_video'][video_idx]
        if video_idx == 0:
            sleep_time = all_userview[user_idx]['sleep_time']     
        else:
            sleep_time = 0            
    else:
        print("All videos have been played")
        return False    
    return view_video, sleep_time



if __name__ == '__main__':
    all_userview = load_all_userview()
    print(load_userview(10,2))

