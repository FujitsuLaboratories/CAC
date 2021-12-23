# CAC同期緩和サンプルプログラム

## 実行環境の準備
- 実行に当たってNVIDIA GPUを搭載したUbuntu OSマシンが必要になります
    - 1台のマシンでも複数台のマシンでも実行可能です
    - 同期緩和はマルチGPU向けの技術ですが、本サンプルプログラムは1 GPU環境でも実行可能です
- 複数台のマシンを使って実行する場合は以下を確認してください
    - マシン間はEthernetもしくはEthernetに加えInfiniBand接続されている
    - マシン間はパスフレーズなしでsshログインできる
    - 実行に必要なライブラリやプログラムは、NFSなどの共有ファイルシステムでマシン間で共有されている

## 環境構築方法（ソフトウェア）
- Open MPI, PyTorch, NVIDIA apex, NVIDIA NCCL、CACライブラリをインストールしてください
- 以下ではpython3.7を仮想環境で構築する場合を示します

### Open MPIのインストール (3.1.6例)
```
$ wget https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.6.tar.bz2
$ tar jxvf openmpi-3.1.6.tar.bz2
$ cd openmpi-3.1.6
$ ./configure --prefix=${HOME}/my-openmpi-3.1.6
$ make
$ make install
```
- インストール先ディレクトリはconfigure時のprefixオプションで指定可能です
- インストール先のディレクトリへのパス設定を実施してください

### miniconda インストール＆セットアップ
```
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash Miniconda3-latest-Linux-x86_64.sh
$ . ~/.bashrc
(base) $ conda config --append channels conda-forge
(base) $ conda config --remove channels defaults
(base) $ conda config --show channels
channels:
  - conda-forge
(base) $ conda update -n base -c defaults conda
```

### python3.7環境構築
```
(base) $ conda create -n py37 python=3.7
(base) $ vi ~/.bashrc
(add a following line at the bottom)
conda activate py37
(base) $ . ~/.bashrc
(py37) $
```

### PyTorch (Torchvision）のインストール
- あらかじめnvidia-smiコマンドなどでCUDAのバージョンを調べておきます
- cudatoolkitのバージョンをCUDAに合わせてください(以下は10.1の場合)
- 複数のCUDAバージョンがインストール済みの場合、利用するバージョンをCUDA_HOME環境変数に指定してください
```
(py37) $ conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
```

### NVIDIA apexライブラリのインストール
```
(py37) $ git clone https://github.com/NVIDIA/apex
# apexを配置
(py37) $ cd apex
# 必須パッケージインストール
(py37) $ pip install -r requirements.txt
# apex　インストール
(py37) $ pip install -v --no-cache-dir --global-option="--pyprof" --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### NVIDIA NCCLライブラリのインストール
- https://github.com/NVIDIA/nccl のREADME.mdなどを参照してください

### CACライブラリのインストール
```
$ git clone https://github.com/FujitsuLaboratories/CAC
$ cd ./CAC
$ python setup.py install
```

## 実行方法
### データセットのダウンロード
```
$ ./download_dataset.sh
```
- dataディレクトリにデータセットがダウンロードされ、展開されます

### 実行用環境変数の設定
- mpi.shをテキストエディタで編集し、実行用環境変数を設定してください
    - DIR変数 ... 実行スクリプトのあるディレクトリを指定します
    - GPUS_PER_NODE ... マシン当たりのGPU数を指定します
    - MASTER_ADDR ... 実行マシンのIPアドレスを指定します。複数台マシンの場合はいずれかのマシンのIPアドレスで構いません
    - MASTER_PORT ... ポート番号を指定します（デフォルトの番号が使用中の場合は他の番号を指定）
- 以下設定例
```
DIR=${HOME}/CAC/cac/relaxed_sync/examples/alexnet_cifar10
GPUS_PER_NODE=2
MASTER_ADDR='172.20.51.33'
MASTER_PORT='29500'
```

- 1GPU環境では、`SINGLE_GPU`変数を1に設定してください
```
SINGLE_GPU=1
```

### 同期緩和の効果実験用の環境変数
- 同期緩和の有効無効や遅いプロセスのシミュレートを切り替える
- mpi.shの以下変数を設定する
    - USE_RELAXED_SYNC
    - USE_SIMULATE
- （例）通常の実行（遅いプロセスのシミュレートなし、同期緩和なし）
```
USE_RELAXED_SYNC=0
USE_SIMULATE=0
```
- （例）遅いプロセスをシミュレート、同期緩和の適用なし
```
USE_RELAXED_SYNC=0
USE_SIMULATE=1
```
- （例）遅いプロセスのシミュレートし、同期緩和の適用あり
```
USE_RELAXED_SYNC=1
USE_SIMULATE=1
```

### 実行
```
$ mpirun --hostfile hosts.txt -np 2 ./mpi.sh
```
- マシン台数に合わせてnpオプションの数を指定します（上記は2台の場合になります。1台であれば`-np 1`になります）
- hosts.txtファイルに実行に使用するマシンのホスト名を記載します

2台（machine1, machine2）の場合：
```
machine1 slots=1
machine2 slots=1
```

1台（machine1）の場合：
```
machine1 slots=1
```


## Copyright
COPYRIGHT Fujitsu Limited 2021
