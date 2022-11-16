import os
import argparse
import numpy as np
import pandas as pd

def generate_graph_seq2seq_io_data(df, x_offsets, y_offsets, add_time_in_day, add_day_in_week, scaler=None):
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    feature_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)
    if add_day_in_week:
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        dow_tiled = dow_tiled / 7
        feature_list.append(dow_tiled)

    data = np.concatenate(feature_list, axis=-1)
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    df = pd.read_hdf(args.traffic_df_filename)
    
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))
    
    num_samples = df.shape[0]
    print('total samples:', num_samples)
    num_train = int(num_samples * 0.6)
    num_val = int(num_samples * 0.8)
    
    x_train, y_train = generate_graph_seq2seq_io_data(df[:num_train], x_offsets, y_offsets, args.tod, args.dow)
    x_val, y_val = generate_graph_seq2seq_io_data(df[num_train: num_val], x_offsets, y_offsets, args.tod, args.dow)
    x_test, y_test = generate_graph_seq2seq_io_data(df[num_val:], x_offsets, y_offsets, args.tod, args.dow)

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="pems-04.h5", help="raw traffic readings")
    parser.add_argument("--output_dir", type=str, default="PEMS-04", help="output directory")
    parser.add_argument("--seq_length_x", type=int, default=12, help="sequence length")
    parser.add_argument("--seq_length_y", type=int, default=12, help="sequence length")
    parser.add_argument("--y_start", type=int, default=1, help="Y pred start")
    parser.add_argument("--tod", type=int, default=1, help="time of day")
    parser.add_argument("--dow", type=int, default=0, help="day of week")

    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        pass
    else:
        os.makedirs(args.output_dir)
    generate_train_val_test(args)
