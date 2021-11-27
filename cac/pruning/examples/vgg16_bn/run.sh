CUDA_VISIBLE_DEVICES='0,1,2,3' python3 main.py --model_path pretrained_cifar10_vgg16_bn.pt --use_gpu --use_DataParallel --pruned_model_path pruned_cifar10_vgg16bn_01.pt > log.log
