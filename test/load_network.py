#coding:utf-8
import os


COOKED_NETWORK_FOLDER = './cooked_traces/'


def load_network(cooked_network_folder=COOKED_NETWORK_FOLDER):
    cooked_files = os.listdir(cooked_network_folder)
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    for cooked_file in cooked_files:
        file_path = cooked_network_folder + cooked_file
        cooked_time = []
        cooked_bw = []
        # print file_path
        with open(file_path, 'rb') as f:
            for line in f:
                parse = line.split()
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]))
        all_cooked_time.append(cooked_time)
        all_cooked_bw.append(cooked_bw)
        all_file_names.append(cooked_file)
    #print(all_cooked_bw[1])
    return all_cooked_time, all_cooked_bw, all_file_names

if __name__ == '__main__':
    all_cooked_time, all_cooked_bw, all_file_names = load_network()
    print(all_cooked_bw[0])    