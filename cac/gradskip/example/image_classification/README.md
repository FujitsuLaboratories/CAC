# Image classification samples

This is a sample of image classification using resnet50 and alexnet.

## How to execute
Run each shell script under `CAC_Lib/example/image_classification/script`.

#### Example of single process execution
```
$ ./example_imagenet.sh <path to imagenet train/val data>
```
#### Example of multi-process execution
```
$ ./mpi.sh 4
```
When executing, rewrite the contents of `./example_imagenet.sh` as necessary.
```
user="<user>"
script_dir=<path to CAC_Lib/example/image_classification/>
log_dir=<log dir>
```
Network models resnet50 or alexnet are available.<br>
It is specified by the execution option `--arch` of `CAC_Lib/example/image_classification/example_imagenet.py`.
#### resnet50
```
python example_imagenet.py ... --arch resnet50 ...
```
#### alexnet
```
python example_imagenet.py ... --arch alexnet ...
```
### Option list
|Option name|Function|Possible values|Default value|
|:--|:--|:--|:--|
|-h, --help|Display help messages|||
|--data-backend|Data backend|`pytorch`, `syntetic`|`pytorch`|
|--arch|Model architecture|`resnet50`, `alexnet`|`resnet50`|
|--model-config|Model configs|`classic`, `fanin`, `grp-fanin`, `grp-fanout`|
|--j|Number of data loading workers|Any value|5|
|--epochs|Number of total epochs to run|Any value|90|
|--batch-size|Mini-batch size per GPU|Any value|256|
|--optimizer-batch-size|Size of a total batch size, for simulating bigger batches using gradient accumulation|Any value|-1|
|--lr|Initial learning rate|Any value|0.1|
|--lr-schedule|Type of LR schedule|`step`, `linear`, `cosine`|`step`|
|--warmup|Number of warmup epochs|Any value|0|
|--label-smoothing|Label smoothing|Any value|0.0|
|--mixup|Mixup alpha|Any value|0.0|
|--momentum|Momentum|Any value|0.9|
|--weight-decay|Weight decay|Any value|1e-4|
|--nesterov|Use nesterov momentum|bool value|false|
|--print-freq|Print frequency (iter)|Any value|10|
|--resume|Path to latest checkpoint|Any string (path)||
|--pretrained-weights|Load weights from here|Any string (path)||
|--fp16|Run model fp16 mode (cannot be used with --amp)|||
|--static-loss-scale|Static loss scale, positive power of 2 values can improve fp16 convergence|Any value|1|
|--dynamic-loss-scale|Use dynamic loss scaling (if supplied, this argument supersedes --static-loss-scale)|||
|--amp|Run model AMP (automatic mixed precision) mode (cannot be used with --fp16)|||
|--local_rank|Local rank of the python process (set in the distributed launcher)|Any value|0|
|--seed|Random seed used for numpy and pytorch|Any value||
|--gather-checkpoints|Gather checkpoints throughout the training, without this flag only best and last checkpoints will be stored|||
|--raport-file|File in which to store JSON experiment raport|Any string|
|--evaluate|Evaluate checkpoint/model|||
|--training-only|Do not evaluate|||
|--no-checkpoints|Do not store any checkpoints, useful for benchmarking|||
|--workspace|Path to directory where checkpoints will be stored|Any string (directory)|`./`|

## Copyright  

COPYRIGHT Fujitsu Limited 2021
