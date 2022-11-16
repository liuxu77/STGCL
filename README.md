# STGCL

This is the official implementation of the paper [When Do Contrastive Learning Signals Help Spatio-Temporal Graph Forecasting?](https://arxiv.org/pdf/2108.11873.pdf)

## Dataset
In the data folder, generate the training, validation and test sets by using:
```
python gen_data.py --input_file pems-0x.h5 --output_dir PEMS-0x
```

Generate the adjacency matrix by using:
```
python gen_adj.py --dataset 0x
```

## Run
To facilitate usage, we put the code of different settings into different folders. For example, joint-graph folder means we use the joing learning scheme, and the contrastive objective is at the graph level.

In the joint-graph/node folders, users can choose to run the pure base model by setting the method to 'pure', or turn on the contrastive learning function by setting the method to 'graph' or 'node'.

In the pretrain-graph/node folders, users need to firstly perform unsupervised training by using u_train.py file, and then utilize s_train.py to conduct the forecasting tasks.


## Reference
Please cite our paper if you use our approach in your work:
```
@inproceedings{liu2022when,
  title={When Do Contrastive Learning Signals Help Spatio-Temporal Graph Forecasting?},
  author={Liu, Xu and Liang, Yuxuan and Huang, Chao and Zheng, Yu and Hooi, Bryan and Zimmermann, Roger},
  booktitle={Proceedings of the 30th International Conference on Advances in Geographic Information Systems},
  year={2022}
}
```
