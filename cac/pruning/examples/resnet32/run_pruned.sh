CUDA_VISIBLE_DEVICES='0' python3 main_pruned.py --model_path ./pruned_cifar10_resnet32.pt --use_gpu --use_DataParallel > log.log
