#coding:utf-8
import os
import csv
import math
import config as parameter
COOKED_VIEWPORT_FOLDER = './tile'

OLD_ROW_MAX = parameter.get_config('OLD_ROW_MAX')
OLD_COL_MAX = parameter.get_config('OLD_COL_MAX')
NEW_ROW_MAX = parameter.get_config('NEW_ROW_MAX')
NEW_COL_MAX = parameter.get_config('NEW_COL_MAX')
fps = parameter.get_config('fps')
video_names = parameter.get_config('video_names')
USER_MAX = parameter.get_config('USER_MAX')


def get_median(data):#中位数
    data.sort()
    half = len(data) // 2
    return (data[half] + data[~half]) / 2

def viewport_to_viewpoint(viewport,row_max,col_max):#提取该viewport的viewpoint
    all_row = []
    all_col = []
    for tile in viewport:
        row = (int(tile)-1) // col_max + 1
        col = (int(tile)-1) % col_max + 1
        all_row.append(row)
        all_col.append(col)
    viewpoint = (get_median(all_row),get_median(all_col))
    return viewpoint



def resize_viewport(viewpoint,t,old_row_max=OLD_ROW_MAX,old_col_max=OLD_COL_MAX,new_row_max=NEW_ROW_MAX,new_col_max=NEW_COL_MAX):#将tile从20*10转变为4*3
	new_row = int(viewpoint[0]*new_row_max/old_row_max)
	new_col = int(viewpoint[1]*new_col_max/old_col_max)
	new_viewpoint = (new_row,new_col)
	new_viewport = []
	length = int(min((math.ceil(new_row_max / 2.0) + 2*int(t)),new_row_max))
	width = int(min((math.ceil(new_row_max / 2.0) + 2*int(t)),new_col_max))
	for i in range(length):
		for j in range(width):
			row = (new_row + (i - int(length / 2.0))) % new_row_max
			col = (new_col + (j - int(width / 2.0))) % new_col_max
			new_viewport.append(int((row)*new_col_max + col))
	new_viewport.sort()
	return new_viewport, new_viewpoint


'''
def resize_viewport(viewport,old_row_max,old_col_max,new_row_max,new_col_max):#将tile从20*10转变为4*3
    new_viewport = []
    for tile in viewport:
        if int(tile) > old_row_max*old_col_max - 1 or int(tile) < 1:
            continue
        row = (int(tile)-1) // old_col_max + 1
        col = (int(tile)-1) % old_col_max + 1
        new_row = math.ceil(row * new_row_max / old_row_max)
        new_col = math.ceil(col * new_col_max / old_col_max)
        new_tile = (new_row-1)*new_col_max + new_col - 1
        new_viewport.append(int(new_tile))
    new_viewport = sorted(list(set(new_viewport)))
    return new_viewport
'''

def load_all_viewport(cooked_viewport_folder=COOKED_VIEWPORT_FOLDER,old_row_max=OLD_ROW_MAX,old_col_max=OLD_COL_MAX,new_row_max=NEW_ROW_MAX,new_col_max=NEW_COL_MAX):
    global all_viewport
    all_viewport = []
    cooked_files = os.listdir(cooked_viewport_folder)
    for cooked_file in cooked_files:
        user_viewport = []
        files = os.listdir(cooked_viewport_folder+'/'+cooked_file)
        for file in files:
            segment_viewport = []
            file_path = cooked_viewport_folder+'/'+cooked_file+'/'+file
            with open(file_path, 'r') as f:
                frames = f.readlines()[1:]
                for frame_idx, frame in enumerate(frames):
                    frame = frame.split(',')[1:]
                    if frame_idx % fps == 0:
                        viewpoint = viewport_to_viewpoint(frame,new_row_max,new_col_max)                    
                        new_viewport, new_viewpoint =  resize_viewport(viewpoint,0)                    
                        segment_viewport.append(new_viewport)
            user_viewport.append(segment_viewport)
        all_viewport.append(user_viewport)
    return all_viewport

def load_viewport(user_idx,video_name,segment_idx,t,new_row_max=NEW_ROW_MAX,new_col_max=NEW_COL_MAX):
    global all_viewport
    user_idx = user_idx % USER_MAX
    viewport = all_viewport[video_names.index(video_name)][user_idx][segment_idx]
    #viewpoint = viewport_to_viewpoint(viewport,new_row_max,new_col_max)
    #new_viewport, new_viewpoint =  resize_viewport(viewpoint,t)
    return viewport

def get_central_viewport(old_row_max=OLD_ROW_MAX,old_col_max=OLD_COL_MAX,new_row_max=NEW_ROW_MAX,new_col_max=NEW_COL_MAX):
    viewport = [n for n in range(0, int(new_row_max*new_col_max))]
    viewpoint = (int(new_row_max/2), int(new_col_max/2))
    return viewport,viewpoint    


if __name__ == '__main__':
    #viewport,viewpoint = load_viewport(1,'landscape',30,COOKED_VIEWPORT_FOLDER)
    #viewport,viewpoint = get_central_viewport()
    all_viewport = load_all_viewport()
    viewport, viewpoint  = load_viewport(1,'coaster',15,0.8)
    viewport, viewpoint = get_central_viewport()
    #new_viewport = resize_viewport(viewpoint,2.8)
    print(viewport, viewpoint)
