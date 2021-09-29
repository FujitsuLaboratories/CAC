# Automatic Pruner Ver 1.0.0

Automatic Pruner is a Python module for pruning neural networks.
This module has the following features.
* The pruning rate of each layer can be determined automatically.
* It can also be applied to convolution layers to which BatchNorm layers are not connected and fully connected layers.

## Requirements

Automatic Pruner requires:
* Python (>= 3.6.7)
* Torch (>= 1.5.0a0+ba48f58)
* Torchvision (>= 0.6.0+cu101)
* Numpy (>= 1.18.2)
* tqdm (>= 4.62.0)

## The directory structure

The directory structure of the Automatic Pruner source is shown below.
```
pruning
  ├── auto_prune.py  (auto-pruner main module)
  └── examples
     ├── AlexNet
     │   ├── alexnet.py
     │   ├── main.py
     │   └── make_model.py
     ├── MLP  (Multi layer perceptron)
     │   ├── main.py
     │   ├── make_model.py
     │   └── mlp.py
     ├── ResNet18
     │   ├── resnet.py
     │   ├── main.py
     │   └── schduler.py
     ├── VGG11
     │   ├── main.py
     │   ├── make_model.py
     │   └── vgg.py
     └── VGG11_bn
         ├── main.py
         ├── schduler.py
         └── vgg_bn.py
```

## How to run the examples

Let's take `AlexNet` as an example. 

### 1. Advance preparation

Perform the following procedure.
* Clone this Github or download the source code set.
* Launch a suitable terminal and change to the `examples/AlexNet` directory.

### 2. Creating a model

Please prepare your own "pre-trained model" for pruning.
Also, set the path for pre-trained model on args.model_path in main.py in each examples.
You can also use model creating scipts in some examples.
```python
python3 make_model.py
```
Pretrained model files will be available soon.

### 3. Run pruning
Execute the following command. 
Use --data to specify the data path.
```bash
# On GPU
python3 main.py --use_gpu --use_DataParallel

# On CPU
python3 main.py

``` 

### 4. Check pruning results

Following output means success.

```bash
(omitted)
===== model: after pruning ==========
DataParallel(
  (module): AlexNet(
    (conv1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (conv2): Conv2d(64, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (conv3): Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv4): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (pool5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (avgpool): AdaptiveAvgPool2d(output_size=(4, 4))
    (drop1): Dropout(p=0.5, inplace=False)
    (fc1): Linear(in_features=4096, out_features=4096, bias=True)
    (drop2): Dropout(p=0.5, inplace=False)
    (fc2): Linear(in_features=4096, out_features=4096, bias=True)
    (fc3): Linear(in_features=4096, out_features=10, bias=True)
  )
)
===== Results =====
Model size before pruning (Byte): 145504179
Model size after pruning  (Byte): 25541488
Compression rate                : 0.824
Acc. before pruning: 90.59
Acc. after pruning : 89.72
Arguments of pruned model:  {'out_ch_conv1': 41, 'out_ch_conv2': 105, 'out_ch_conv3': 158, 'out_ch_conv4': 131, 'out_ch_conv5': 131, 'out_ch_fc1': 1678, 'out_ch_fc2': 1342}
```
note) The number of channels in each layer of the model displayed in `model: after pruning` and each numerical value of` Results` may differ depending on the execution environment.

Other samples stored in examples directory can be executed same as AlexNet.

## How to prune user-created model

note)There are limits to the configuration of models that can be pruned (see `Limitations`)

### 1. Change model definition

auto_prune assumes that the model is defined as a class` that inherits from `torch.nn.Module.
In order to apply auto_pruner, you need to make the following changes to the user-defined class.
* Add the argument `out_channels` of the torch.nn.Conv2d (or Conv1d) layer and the argument` out_features` of the torch.Linear layer to the argument of the `__init __` method.This allows you to specify out_channels and out_features values for each layer when instantiating.
* The number of outputs in the final layer is fixed. In the case of CIFAR10, the number of outputs in the final layer is 10.
* Adjust the input / output of layers that are not subject to pruning in accordance with the above changes.

### 2. model_info settings

model_info is the configuration information of the pruning network. auto_pruner performs pruning based on this information.
The AlexNet sample model_info is shown below.
```python
from collections import OrderedDict
# Model information for pruning
model_info = OrderedDict(conv1={'arg': 'out_ch_conv1'},  # layer 1
                         conv2={'arg': 'out_ch_conv2'},
                         conv3={'arg': 'out_ch_conv3'},
                         conv4={'arg': 'out_ch_conv4'},
                         conv5={'arg': 'out_ch_conv5'},
                         fc1={'arg': 'out_ch_fc1'},
                         fc2={'arg': 'out_ch_fc2'},
                         fc3={'arg': None})  # The output of last layer must be set None
```


* As described above, the relation of each layer name and its output argument name is defined by modelinfo.
* Each layer described in modelinfo must be described in the order of network forward propagation.
* The layer to be described is `torch.nn.Conv2d (or Conv1d)`, `torch.nn.Linear`, `torch.nn.BatchNorm2d (or BatchNorm1d)`.
Do not describe Dropout layer, ReLu layer, etc.
* Each layer name (instance variable name) defined in the model class is entered in the key of modelinfo.

However, when multiple layers are defined together using `torch.nn.Sequential` etc., each layer name is named automatically. 
For example, if multiple layers are defined together as shown below,
  ```python
  import torch.nn as nn
  self.classifier = nn.Sequential(nn.Linear(out_ch_conv8 * 1 * 1, out_ch_fc1),  # 1
                                  nn.ReLU(True),  # Not applicable to model_info
                                  nn.Dropout(),  # Not applicable to model_info
                                  nn.Linear(out_ch_fc1, out_ch_fc2),  # 2
                                  nn.ReLU(True),  # Not applicable to model_info
                                  nn.Dropout(),  # Not applicable to model_info
                                  nn.Linear(out_ch_fc2, num_classes))  # 3(last layer)
  ```

  model_info should be specified as follows

  ```python
  model_info = OrderedDict()
  model_info['classifier.0'] = {'arg': 'out_ch_fc1'}  # 1
  model_info['classifier.3'] = {'arg': 'out_ch_fc2'}  # 2
  model_info['classifier.6'] = {'arg': None}  # 3(last layer)
  ```
* For the value of each key of OrderedDict, specify `{'arg': output argument name of the layer}`.
However, since the output of the final layer is not subject to pruning, its output argument should be `None`.

#### For networks with residual connections
If there is a residual connection immediately before the target layer, specify the layer name immediately before it with `'prev'`key as shown below.

As an example, a part of the network configuration diagram of sample `ResNet18` and model_info are described like this.

<img src="images/res18.png" width="900">

In this case, there are a residual connections (# 1, # 2) just before the `'layer1.1.conv1'` layer and the `'layer2.0.conv1'` layer.
* The layer names immediately before the'layer1.1.conv1'layer are'bn1'and'layer1.0.bn2'.
* The layer names immediately before the'layer2.0.conv1'layer are'bn1',' layer1.0.bn2' and'layer1.1.bn2'.
When this is reflected in model_info, it becomes as follows.

```python
# Model information for pruning
model_info = OrderedDict()
(omitted)
model_info['layer1.1.conv1'] = {'arg': 'out_ch_l1_1_1', 'prev': ['bn1', 'layer1.0.bn2']}  # 1
(omitted)
model_info['layer2.0.conv1'] = {'arg': 'out_ch_l2_0_1', 'prev': ['bn1', 'layer1.0.bn2', 'layer1.1.bn2']}  # 2
(omitted)
```
See sample `examples \ ResNet18` for details.  


### 3. Execution of auto-prune function

First, import the `auto_prune` function from` auto_prune.py`. 

An example of import
```python
from cac import auto_prune
```

Next, pass the arguments that match the user's model to the `auto_prune` function and execute it.  
An example of using auto_prune is described below.

```python
weights, Afinal, n_args_channels = auto_prune(AlexNet, model_info, weights, Ab,
                                              train_loader, val_loader, criterion)
```
For more information about model_info, refer to the section of `model_info settings`.

See the `Docstring` section for details on arguments and return values.

#### Docstring

The Docstring of the auto_pruning function is shown below.
```python
"""Automatically decide pruning rate

Args:
    model_class (-): User-defined model class
    model_info (collections.OrderedDict): Model information for auto_prune
    weights_before (dict): Weights before purning
    acc_before (float or numpy.float64): Accuracy before pruning
    train_loader (torch.utils.data.dataloader.DataLoader): DataLoader for
                                                           training
    val_loader (torch.utils.data.dataloader.DataLoader): DataLoader for
                                                         validation
    criterion (torch.nn.modules.loss.CrossEntropyLoss, optional):
                                             Criterion. Defaults to None.
    optim_type (str, optional): Optimizer type. Defaults to 'SGD'.
    optim_params (dict, optional): Optimizer parameters. Defaults to None.
    lr_scheduler(-): Scheduler class. Defaults to None.
    scheduler_params(dict, optional): Scheduler parameters. Defaults to None
    update_lr(str, optional): 'epoch': Execute scheduler.step()
                                       for each epoch.
                              'step': Execute scheduler.step()
                                      for each training iterarion.
                              Defaults to 'epoch'
    use_gpu (bool, optional): True : use gpu.
                              False: do not use gpu.
                              Defaults to False.
    use_DataParallel (bool, optional): True : use DataParallel().
                                       False: do not use DataParallel().
                                       Defaults to True.
    loss_margin (float, optional): Loss margin. Defaults to 0.1.
    acc_margin (float, optional): Accuracy margin. Defaults to 1.0.
    trust_radius (float, optional): Initial value of trust radius
                                    (upper bound of 'thresholds').
                                    Defaults to 10.0.
    scaling_factor (float, optional): Scaling factor for trust raduis.
                                      Defaults to 2.0.
    rates (list, optional): Candidates for pruning rates.
                            Defaults to None.
    max_iter (int, optional): Maximum number of pruning rate searching.
    calc_iter (int, optional): Iterations for calculating gradient
                               to derive threshold.
                               Defaults to 100.
    epochs (int, optional): Re-training duration in pruning rate search.
    model_path (str, optional): Pre-trained model filepath.
                                Defaults to None.
    pruned_model_path (str, optional): Pruned model filepath.
                                       Defaults to './pruned_model.pt'.
    residual_info (collections.OrderedDict, optional): Information on
                                                       residual connections
                                                       Defaults to None.
    residual_connections (bool, optional): True: the network has
                                                  residual connections.
                                           False: except for the above.
                                           Defaults to False.

Returns:
    weights_before(dict): Weights after purning
    final_acc(float): Final accuracy with searched pruned model
    n_args_channels: Final number of channels after pruning
"""
```
note) For networks with residual connections such as ResNet18, the argument residual_connections must be set to True.

### 4. How to use the model after pruning

As shown in the example below, instantiate the model with the number of channels after pruning as an argument and load state_dict.
```python
from alexnet import AlexNet
model = AlexNet(**n_args_channels)  # n_args_channels is a return value of auto_prune function
model.load_state_dict(torch.load(pruned_model_path), strict=True)
```

## Tips

### How to rewrite the key name of the trained model

When changing the key name such as the weight of the trained model, save the weight of the model once with state_dict ().  
At this time, state_dict is just an OrderedDict type, and the key can be changed as needed.
You can check the model key with `model.state_dict (). Keys ()`.  

### Exclude certain layers from pruning

You can exclude specific layers from pruning by adding `'prune': False` to model_info.
For example, if you specify the following in AlexNet model_info, `conv1, conv3, conv5, fc1` will be excluded from pruning.
```python
model_info = OrderedDict(conv1={'arg': 'out_ch_conv1', 'prune': False},
                         conv2={'arg': 'out_ch_conv2'},
                         conv3={'arg': 'out_ch_conv3', 'prune': False},
                         conv4={'arg': 'out_ch_conv4'},
                         conv5={'arg': 'out_ch_conv5', 'prune': False},
                         fc1={'arg': 'out_ch_fc1', 'prune': False},
                         fc2={'arg': 'out_ch_fc2'},
                         fc3={'arg': None})
```

### When the compression ratio of the model is low

If the model compression ratio is low with the default settings, consider the following measures.

* Increase the number of epochs
* Increase trust_radius
* Increase acc_margin(Be careful as it will lead to a decrease Acc after pruning.)

## Limitations  

* The layers targeted for pruning are only the torch.nn.Conv2d (or Conv1d)) layer and the torch.nn.Linear layer.  
However, the num_features of BatchNorm2d (or BatchNorm1d) will also be reduced.

* The networks that have been confirmed to work are as follows
  * Serial networks
    * AlexNet(examples/AlexNet)
    * VGG11 without batchnorm layer(exmples/VGG11)
    * VGG11 with batchnorm layer(exmples/VGG11_bn)
    * Neural network consisting only from fully connected layers(examples/MLP)
  * Neural network with residual connections
    * ResNet18(examples/ResNet18)

* Depending on how the model is defined, pruning execution may fail.  

## Cautions

* The attached sample is for confirming the effect of this module, and may not be suitable for practical use.
* The commands described in this Readme depend on the execution environment.

## Copyright  

COPYRIGHT Fujitsu Limited 2021