# ----------------------------------------------------------------
# Find patterns of infections of COVID-19 among countries and regions
# Written by Haiyang Jiang
# Mar 19th 2020
# ----------------------------------------------------------------

import glob, os
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances

# ============================================================
# config
metric = lambda a,b: sum(abs(a - b)) # l1 norm
NORMALIZE = True # when calculate distances scale input vectors to have max=1
PARAMETRIC_MDS = True # performs better than non-parametric MDS

file = './results/l1_normalized_parametricMDS'
fig_file = file + '.png'
text_file = file + '_group_info.txt'
SHOW_MDS_FIG = True # if false, then save to local file

data_files = glob.glob('./data/*')
np.random.seed(ord('c') + 137)
# ============================================================


def read_in(file_name):
    with open(file_name) as f:
        text = f.readlines()
    numbers = np.array([float(line.strip().split('\t')[1]) for line in text])
    return numbers


def slide_window_distance(x, y):
    '''
        find smallest possible difference between vector x and y
        by sliding the shorter one through the longer one
    '''
    if len(x) < len(y):
        short = x
        long = y
    else:
        short = y
        long = x
    min_dist = metric(long[:len(short)], short)
    index = 0
    for i in range(1, len(long) - len(short) + 1):
        dist = metric(long[i: i + len(short)], short)
        if dist < min_dist:
            min_dist = dist
            index = i
    return min_dist, index


def calc_dist(data):
    '''
        calculate pairwise distance
    '''
    N = len(data)
    pairwise_distance = np.zeros([N, N])
    index = np.zeros([N, N], dtype=int)
    for i in range(N):
        for j in range(i + 1, N):
            if NORMALIZE: # scale maximum value to 1
                distance, idx = slide_window_distance(data[i][1] / max(data[i][1]), data[j][1] / max(data[j][1]))
            else:
                distance, idx = slide_window_distance(data[i][1], data[j][1])
            pairwise_distance[i, j] = pairwise_distance[j, i] = distance
            index[i, j] = index[j, i] = idx
    return pairwise_distance, index


def find_and_pop(indices, index):
    for i in range(len(indices)):
        if indices[i][0] == index:
            return indices.pop(i)


def grouping(indices):
    new_indices = indices[:]
    visited = set()
    groups = []
    while new_indices:
        idx, next = new_indices.pop()
        visited.add(idx)
        while next not in visited and new_indices:
            idx, next = find_and_pop(new_indices, next)
            visited.add(idx)
        added = True
        while added:
            added = False
            for i in range(len(new_indices) - 1, -1, -1):
                indices = new_indices[i]
                if indices[1] in visited:
                    added = True
                    visited.add(indices[0])
                    _ = new_indices.remove(indices)
        groups.append(np.array(list(visited)))
        visited = set()
    return groups


def get_grouping(distance):
    neighbors = []
    for i in range(distance.shape[0]):
        temp = distance[i, :].copy()
        temp[i] = temp.max()
        j = np.argmin(temp)
        neighbors.append([i, j])
    return grouping(neighbors)


def country_index(country, data):
    for i in range(len(data)):
        if data[i][0] == country:
            return i
    return -1


def plot_group(group, data, index, log_scale=True):
    max_len = 0
    idx_longest = 0
    for i in group:
        d = data[i]
        if len(d[1]) > max_len:
            longest = d[0]
            max_len = len(d[1])
            idx_longest = i
    all_days = np.array(range(len(data[idx_longest][1])))
    for i in group:
        country = data[i][0]
        if country == longest:
            if log_scale:
                plt.plot(all_days, np.log(data[idx_longest][1]), label=longest)
            else:
                plt.plot(all_days, data[idx_longest][1], label=longest)
            continue
        idx = index[i, idx_longest]
        days = np.array(range(idx, idx + len(data[i][1])))
        if log_scale:
            plt.plot(days, np.log(data[i][1]), label=country)
        else:
            plt.plot(days, data[i][1], label=country)
    plt.legend(loc='best')


def main():
    data = []
    for file in data_files:
        data.append([os.path.basename(file).split('.')[0], read_in(file)])
    distance, index = calc_dist(data)
    
    embedding = MDS(n_components=2, metric=PARAMETRIC_MDS, dissimilarity='precomputed')
    
    X_transformed = embedding.fit_transform(distance)
    
    fig = plt.figure()
    for i in range(len(data)):
        plt.plot(X_transformed[i, 0], X_transformed[i, 1], 'x')
        plt.text(X_transformed[i, 0], X_transformed[i, 1], data[i][0])
    plt.title('Feature Coordinates of COVID-19 Infection Pattern in Each Country')
    plt.xlabel('Dimension 1 in Transformed Space')
    plt.ylabel('Dimension 2 in Transformed Space')
    
    if SHOW_MDS_FIG:
        plt.show()
    else:
        plt.savefig(fig_file, dpi=300)
        plt.close()
    
    # write grouping info to text file
    nation_list = np.array([d[0] for d in data])
    groups = get_grouping(distance)
    with open(text_file, 'w') as f:
        for group in groups:
            f.write(', '.join(nation_list[group]) + '\n')
    
    for i, group in enumerate(groups):
        plt.figure()
        plot_group(group, data, index, log_scale=False)
        plt.title('Group ' + str(i + 1))
        plt.xlabel('Days')
        plt.ylabel('# of Infections')
    plt.show()
    
    for i, group in enumerate(groups):
        plt.figure()
        plot_group(group, data, index)
        plt.title('Group ' + str(i + 1))
        plt.xlabel('Days')
        plt.ylabel('log(# of Infections)')
    plt.show()


if __name__ == "__main__":
    main()