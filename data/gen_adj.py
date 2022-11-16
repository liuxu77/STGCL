import pickle
import argparse
import numpy as np
import pandas as pd

def get_adjacency_matrix(dataset, sdistance_df, sensor_id_to_ind, normalized_k):
    print('check duplicate num', sum(distance_df.duplicated()))
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf

    # Fills cells in the matrix with distances
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            print('warning')
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    # Calculates the standard deviation as theta
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))

    # Sets entries that lower than a threshold to zero for sparsity
    adj_mx[adj_mx < normalized_k] = 0
    print('check none zero entries after normalizing', np.count_nonzero(adj_mx))
    return adj_mx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='04', help='dataset name')
    parser.add_argument('--normalized_k', type=float, default=0.1, help='for controlling sparsity')
    args = parser.parse_args()
    
    # Read file and check self loop
    distance_filename = 'PEMS' + args.dataset + '.csv'
    distance_df = pd.read_csv(distance_filename, dtype={'from': 'str', 'to': 'str'})
    print('distance df shape', distance_df.shape)
    t = distance_df['from'] == distance_df['to']
    print('check self loop num', np.sum(t))
    if np.sum(t) != 0:
        for i in range(len(t)):
            if t[i] == True:
                print('index', i)
                distance_df = distance_df.drop([i])
        print('check self loop num', np.sum(distance_df['from'] == distance_df['to']))
    
    # Builds sensor id to index map
    sensor_id_to_ind = {}
    sensor_ids = distance_df['from'].tolist() + distance_df['to'].tolist()
    sensor_ids = list(set(sensor_ids))
    sensor_ids = [int(i) for i in sensor_ids]
    print('check sensor stat', len(sensor_ids), min(sensor_ids), max(sensor_ids))

    sensor_ids = [str(i) for i in range(len(sensor_ids))]
    for i, sensor_id in enumerate(sensor_ids):
            sensor_id_to_ind[sensor_id] = i
     
    # Generate adjacency matrix
    adj_mx = get_adjacency_matrix(args.dataset, distance_df, sensor_id_to_ind, args.normalized_k)
    adj_mx += + np.eye(len(sensor_ids))
    print('check sum of adj mx', np.sum(adj_mx))
    print('total none zero entries', np.count_nonzero(adj_mx))
    
    # Save to pickle file
    output_filename = 'adj_mx_' + args.dataset + '.pkl'
    with open(output_filename, 'wb') as f:
        pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f, protocol=2)
