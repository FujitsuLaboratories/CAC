CUDA_VISIBLE_DEVICES='0,1' python3 main.py --model_path ./pretrained_cifar10_resnet18.pt --use_gpu --use_DataParallel --pruned_model_path pruned_cifar10_resnet18_01.pt> log.log
