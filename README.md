# What's CAC

This repository contains a library of Content-Aware Computing (CAC) by Fujitsu.<br>
CAC is a software technology that aims at easy, high-speed, lightweight, and accurate deep learning processing.

# Contents

## 1. Gradient-Skip

Gradient-Skip is an approach for CNNs to skip backward calculations for layers that enouch converged.<br>
This reduces calculations in backward and communications of gradient.<br>
You can use Gradient-Skip by simply replacing the optimizer with our SGD.<br>

[Python Source](https://github.com/FujitsuLaboratories/CAC/tree/main/cac/gradskip)

[Example](https://github.com/FujitsuLaboratories/CAC/tree/main/cac/gradskip/example/image_classification)

## 2. Automatic Pruning

Automatic Pruning is a pruning tool for neural networks, which can determine the pruning rate of each layer automatically.<br>

[Python Source](https://github.com/FujitsuLaboratories/CAC/tree/main/cac/pruning)

[Example](https://github.com/FujitsuLaboratories/CAC/tree/main/cac/pruning/examples)

## 3. Synchronous-Relaxation
Relaxed Synchronization technique removes slow processes from the group of distributed training and prevent limiting overall training speed due to slow processes. <br>

[Python Source](https://github.com/FujitsuLaboratories/CAC/tree/main/cac/relaxed_sync)

[Example](https://github.com/FujitsuLaboratories/CAC/tree/main/cac/relaxed_sync/examples/imagenet)

# Requirements

Python 3.7 or later

CUDA 10 or later

PyTorch 1.6 or later

Apex

# Quick Start

### Linux

```
pip install --no-cache-dir cac
```
### When download the code by using `git clone`

```
git clone https://github.com/FujitsuLaboratories/CAC.git
cd ./CAC             # move to the directory where 'setup.py' is located.
pip install -e .     # execute `pip install` after moving the directory. 
```
