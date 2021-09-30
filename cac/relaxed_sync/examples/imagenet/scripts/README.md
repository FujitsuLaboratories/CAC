## Scripts for runinng training code (main_amp.py)

* examples/imagenet/scripts/mpi.sh
```
./mpi.sh [nnodes] [arch] [bs] [learning_rate] [opt_level] [base_data_dir]
```

|param|val    |
|------|-------|
|nnodes|Number of compute nodes|
|arch|Model architecture|
|bs|Batch size|
|learning_rate|Learning rate|
|opt_level|Optimization level (AMP) [O0,O1,O2,O3]|
|base_data_dir|Base directory for ImageNet(ILSVRC2012) dataset|

* examples/imagenet/scripts/pytorch_imagenet.sh
    * Modify parameters for your environment

```
nproc_per_node=4
epoch=3
```
|param|val    |
|------|-------|
|nproc_per_node|Number of processes per computer nodes|
|epoch|Number of epochs|

### How to run

```
./mpi.sh [nnodes] [arch] [bs] [learning_rate] [opt_level] [base_data_dir]
```

## Copyright  

COPYRIGHT Fujitsu Limited 2021
