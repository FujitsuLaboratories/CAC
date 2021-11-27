CUDA_VISIBLE_DEVICES='0,1' python3 main_pruned.py --model_path pruned_mnist_mlp.pt --use_gpu --use_DataParallel > log.log
