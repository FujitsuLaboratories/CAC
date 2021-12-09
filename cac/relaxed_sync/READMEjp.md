# 同期緩和
同期緩和技術は、分散並列学習のグループから遅いプロセスを削除し、学習速度の低下を防止する技術です。

データ並列のような分散学習では、各プロセスの学習結果の集約のために通信が必要であり、遅いプロセスがひとつでもあると、同期待ちが発生するためパフォーマンスが低下します。同期緩和は学習結果の集約から遅いプロセスを削除して残りのプロセスの学習結果を使用することでパフォーマンスの低下を防止します。

## 1. 使用方法

同期緩和を使用する場合に必要なコードの修正方法について説明します。具体的な例としてはmain_amp.pyを参照してください。


### 1) RelaxedSyncDistributedDataParallelをインポート

DDPとしてcacのRelaxedSyncDistributedDataParallelを使用します。

    from cac.relaxed_sync import RelaxedSyncDistributedDataParallel as DDP

### 2) modelにDDP(RelaxedSyncDistributedDataParallel)を通し、パラメータrelaxed_sync_thresholdを設定

    model = DDP(self.model, relaxed_sync_threshold=self.relaxed_sync_threshold, relaxed_sync_mode_threshold=self.relaxed_sync_mode_threshold)

|パラメータ|機能|
|------|-------|
|relaxed_sync_threshold|このパラメータは、ターゲットとするプロセスを削除する判断の閾値です。例えばrelaxed_sync_thresholod=2.0のとき、平均プロセス速度より2.0倍遅いプロセスが削除されます。|
|relaxed_sync_mode_threshold|このパラメータは、プロセス数がどの程度減少したらtrain_loaderを再構築するかの閾値です。 例えばrelaxed_sync_mode_threshold=0.5のとき、半分以上のプロセスが削除されるとtrain_loaderが再構築されます。|
|simulate_slow_process|このパラメータは同期緩和の効果の確認に使用します。1に設定すると遅いプロセスが模擬されます。デフォルトでは10エポックから導入されます。|

### 3) train_loaderとval_loaderの作成

PyTorchのDataLoaderの代わりに、cacのRelaxedSyncDistributedDataParallelにより提供されるTrainDataLoaderとValDataLoaderを使用します。

    train_loader = model.TrainDataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=collate_fn)
    val_loader = model.ValDataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=val_sampler,
        collate_fn=collate_fn)

### 4) lossとaccuracyの算出

lossとaccuracyの算出に、cacのDDPのcalc_reduced_tensor(), calc_prec()を使用します。

* lossの集約の例
    ```
    reduced_loss = model.calc_reduced_tensor(loss.data)
    ```

* loss, prec1, prec5の集約の例
    ```
    reduced_loss, prec1, prec5 = model.calc_prec(loss.data, prec1, prec5)
    ```

### 5) 学習ループでset_relaxed_pg()をコール

毎Epoch開始時点で、set_relaxed_pg()をコールする。こうすることで遅いプロセスがあるかどうか、学習ループから取り除くかどうかを判定する。

```
model.set_relaxed_pg(epoch, min_num_processes=min_num_processes)
```
同期緩和は、2)のrelaxed_sync_thresholdに従って遅いプロセスを判定し、遅いプロセスを取り除いた新たな学習プロセスグループを作成する


パラメタのmin_num_processesは、同期緩和によって生成される新たなプロセスグループの最小数を判定する。
例えば、以下のように初期プロセスの半分を指定すれば、たとえ遅いプロセスが残っていても、最大で半分のプロセスだけ削除される。

```
min_num_processes = torch.distributed.get_world_size() / 2
```

### 6) train_loader, val_loaderの再構築

各Epochの最初に、同期緩和によりプロセス数が変わった時にrearrange_data_loadersをコールし、torain_loader, val_loaderを必要に応じて再アレンジします。

同期緩和によりプロセス鵜が変わった時、val_loaderはプロセス数に応じて再構築され、train_loaderは残りプロセス数が初期プロセス数のrelaxed_sync_mode_threshold倍になった時に再構築されます。

```
train_loader, val_loader = model.rearrage_data_loaders(train_loader, val_loader)
```

### 7) Learning rateのコーディネート

学習プロセス数が変わった時、Learning rate(lr)をコーディネートしたい場合には、adjust_lr_by_procsをコールします。
Learning rateはプロセス数が変わって初期プロセス数のrelaxed_sync_mode_threshold倍を下回ったら調整されます。

```
lr = model.adjust_lr_by_procs(init_lr)
```

### 8) 終了処理

main functionに戻るときはfinalize()をコールします。これは削除されたプロセスが終了することを通知するために必要です。

```
model.finalize()
```


## 2. 環境構築
### 動作確認済みのバージョン

```
(py37) $ python --version
Python 3.7.10
(py37) $ pip list
Package             Version
------------------- -------------------
apex        0.1
cac-lib     0.0.1
numpy       1.21.2
olefile     0.46
Pillow      8.3.2
pip         21.2.4
setuptools  58.0.4
torch       1.6.0
torchvision 0.7.0
wheel       0.37.0
```


### Minicondaのインストールとセットアップ

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

### Python 3.7 仮想環境の構築

```
(base) $ conda create -n py37 python=3.7
(base) $ vi ~/.bashrc
(add a following line at the bottom)
conda activate py37
(base) $ . ~/.bashrc
(py37) $
```


### 必要なパッケージのインストール

* PyTorch, torchvisionのインストール

```
(py37) $ conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
```


* NVIDIA APEXのインストール

```
# APEXのダウンロード
(py37) $ git clone https://github.com/NVIDIA/apex
# apexディレクトリに移動
(py37) $ cd apex
# 必要なパッケージのインストール
(py37) $ pip install -r requirements.txt
# APEXのインストール
(py37) $ pip install -v --no-cache-dir --global-option="--pyprof" --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# (then check “Successfully installed apex-0.1” is output at middle of the console output)
```

## Copyright  

COPYRIGHT Fujitsu Limited 2021
