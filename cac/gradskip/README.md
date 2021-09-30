# Gradient-Skip

Gradient-Skip is an approach for CNNs to skip backward calculations for layers that enouch converged.<br>
This reduces calculations in backward and communications of gradient.<br>

## Modify learning script
You can use Gradient-Skip by simply replacing the optimizer with our SGD.<br>
```diff
+ import cac
… 
- optimizer = torch.optim.SGD(
+ optimizer = cac.gradskip.SGD(
              model.parameters(),
              lr=lr,
              weight_decay=weight_decay,
              momentum=momentum)
```

### Gradient-Skip operation method
Gradient-Skip has two operation methods, `Manual` and `Auto`.

|Operation method|Summary|
|:--|:--|
|`Manual`|Method for manually specify the layer number and iteration(*) to skip the learning.|
|`Auto`|Method for automatically determining a layer and timing to skip the learning based on weight variance of each layer.|

(*)In this case, iteration refers to the number of calls to optimizer.

In the initial state of Gradient-Skip, the `Auto` method using the default (recommended) value as a threshold value operates.<br>
If you want to change the method from `Auto` to `Manual`, please see the environment variables shown in the next section.

### Setting environment variables for Gradient-Skip
There are environment variables specific to the `Manual` or `Auto` method and common environment variables.<br>
When `CAC_STOP_LAYER_NUM` and `CAC_STOP_LAYER_ITR` are set, the `Manual` method is used. If the environment variable for the `Auto` method is set at the same time, the manual method takes precedence.

|Environment variables|Operation method|Default (Recommended) value|Function|
|:--|:-:|:-:|:--|
|CAC_STOP_LAYER_NUM|`Manual`|-|Specify the layer to stop the operation.|
|CAC_STOP_LAYER_ITR|`Manual`|-|Specify the iterlation to stop the operation.|
|CAC_VAR_START_ITR|`Auto`|5000|Specify the iteration to start the stop decision of the layer.|
|CAC_VAR_START_THR|`Auto`|0.95|In the first stop target layer, specify the value of the weight variance that will be the threshold for stopping the layer.|
|CAC_VAR_MT_COUNT_THR|`Auto`|5|In each stop target layer, specify the number of times the weight variance peak is updated, which is the threshold for determining that the weight variance of a layer is changing to a mountain shape.|
|CAC_VAR_MT_THR|`Auto`|0.96|In the stop target layer where the weight variance is changing to a mountain shape, specify the value of the weight variance that will be the threshold for stopping the layer.|
|CAC_VAR_SLOPE_THR|`Auto`|0.98|In the stop target layer where the weight variance is changing to a slope type, specify the value of the weight variance that will be the threshold for stopping the layer.|
|CAC_VAR_SAMPLES|`Auto`|200|Specifies the interval between iterations to make a stop decision for the layer.|
|CAC_BRAKING_DISTANCE|common|0|Specify the period (iteration) during which Braking Distance decreases the learning rate.|

#### Example of setting environment variables in `Manual` method
```bash
export CAC_STOP_LAYER_NUM=10,20
export CAC_STOP_LAYER_ITR=0,100
```
#### Example of setting environment variables in `Auto` method
```bash
export CAC_VAR_START_ITR=5000
export CAC_VAR_START_THR=0.95
export CAC_VAR_MT_COUNT_THR=5
export CAC_VAR_MT_THR=0.96
export CAC_VAR_SLOPE_THR=0.98
export CAC_VAR_SAMPLES=200
export CAC_BRAKING_DISTANCE=300  # Set only when using Braking Distance
```
## Copyright  

COPYRIGHT Fujitsu Limited 2021
