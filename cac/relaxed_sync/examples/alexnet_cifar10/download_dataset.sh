#!/usr/bin/env bash
# COPYRIGHT Fujitsu Limited 2021

mkdir data
curl -o data/cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar zxf ./data/cifar-10-python.tar.gz -C ./data
