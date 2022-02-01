# auto-pruner Ver 0.0.0
(江藤コメント:↑は説明のための仮名です、正式名が決まりましたら差し替えてください。)

auto-prunerは、ニューラルネットワークをpruningするためのPythonモジュールである。  
このモジュールは、以下の特長を持つ。
* 各レイヤのpruning率を自動的に決定することができる。
* BatchNormレイヤが接続されていない畳込みレイヤや、全結合レイヤにも適用できる。

(江藤コメント:↑の特長は、坂井さんに頂いた資料からの抜粋です。)

## Requirements

auto-pruner requires:
* Python (>= 3.6.7)
* Torch (>= 1.5.0a0+ba48f58)
* Torchvision (>= 0.6.0+cu101)
* Numpy (>= 1.18.2)

(江藤コメント:↑mattermostでご相談させていただいたように坂井さん環境で動作確認ができたバージョンに更新をお願いいたします)

## ディレクトリ構成

auto-prunerのソースのディレクトリ構成を以下に示す。
```
cac_pruning
  ├── auto_prune.py  (auto-pruner本体)
  └── examples  (サンプル)
     ├── AlexNet
     │   ├── alexnet.py
     │   ├── main.py
     │   └── make_model.py
     ├── MLP  (マルチレイヤパーセプトロン)
     │   ├── main.py
     │   ├── make_model.py
     │   ├── mlp.pt
     │   └── mlp.py
     ├── ResNet18
     │   ├── resnet.py
     │   ├── resnet18.pt
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

## サンプルの実行方法

サンプルの`AlexNet`を例に説明する。  

### 1. 事前準備

以下の手順を実施する。
* 本GithubをClone、もしくはソースコード一式をダウンロードする。  
* 適当なターミナルを起動し、`examples\AlexNet`ディレクトリに移動する。  

### 2. モデルの作成

(江藤コメント:重みの格納場所が決まったら本項を重みの読み込みの説明に書き換え、make_model.pyも削除して良いかもしれません。)  

モデルの作成が未実施の場合、以下のコマンドを実行して、モデルを作成しておく。  
```bash
python3 make_model.py
```   

### 3. pruningの実行  
ターミナルで、以下のコマンドを実行する。  
```bash
# On GPU
python3 main.py --use_gpu --use_DataParallel

# On CPU
python3 main.py

# データのパスを指定する場合は、--dataを使用すること。 
``` 
  
### 4. pruning結果の確認
ターミナルに下記のように出力されれば、実行完了。  

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
注) `model: after pruning`に表示されるモデルの各レイヤのチャネル数や`Results`の各数値は、実行環境等により異なることがある。

`examples`ディレクトリに格納している他のサンプルも、AlexNetと同様の手順で実行可能である。  
ただし、`VGG11_bnとResNet18`のサンプルにはmake_model.pyが存在しないため、  
重みのファイルを`hogehoge`からダウンロードする必要がある。  
(江藤コメント:↑重みの提供方法が決まったら記載を変更してください)

## ユーザーモデルのpruning方法

注) pruning可能なモデルの構成には制限がある(`Limitations`の項を参照)

### 1. モデル定義の変更  

auto_pruneでは、モデルが`torch.nn.Moduleを継承したclass`として定義されていることを前提とする。  
auto_prunerを適用するためには、ユーザーが定義したclassに対して以下の変更を行う必要がある。  
* torch.nn.Conv2d(or Conv1d)レイヤの引数`out_channels`とtorch.Linearレイヤの引数`out_features`を`__init__`メソッドの引数にする。  
これにより、インスタンス化の際に各レイヤのout_channelsとout_featuresの値を指定できるようになる。
* 最終レイヤの出力数は固定とする。CIFAR10の場合は、最終レイヤの出力数は10である。  
* 上記の変更に合わせて、pruning対象外のレイヤの入出力を変更する。  

### 2. model_infoの設定

model_infoは、pruningするネットワークの構成情報である。auto_prunerは、この情報を元にpruningを行う。  
以下にAlexNetのサンプルのmodel_infoを示す。
```python
from collections import OrderedDict
# Model information for pruning
model_info = OrderedDict(conv1={'arg': 'out_ch_conv1'},  # 1レイヤ
                         conv2={'arg': 'out_ch_conv2'},
                         conv3={'arg': 'out_ch_conv3'},
                         conv4={'arg': 'out_ch_conv4'},
                         conv5={'arg': 'out_ch_conv5'},
                         fc1={'arg': 'out_ch_fc1'},
                         fc2={'arg': 'out_ch_fc2'},
                         fc3={'arg': None})  # 最終レイヤの出力はNoneとしておく
```

* 上記のように、ネットワークの各レイヤ名とその出力引数名との対応を、`OrderedDict`で定義する。  
OrderedDictは、Python標準ライブラリの`collections`に含まれるclassである。  
* OrderedDictに記載する各レイヤは、ネットワークの順伝播の順に記述しなければならない。  
これによって、各レイヤの接続順を表現する。
* 記述対象のレイヤは、`torch.nn.Conv2d(or Conv1d), torch.nn.Linear, torch.nn.BatchNorm2d(or BatchNorm1d)`である。  
`DropoutレイヤやReLuレイヤ等は記述しないこと。`
* OrderedDictのkeyには、モデルクラスで定義した各レイヤ名(インスタンス変数名)が入る。  
ただし、`torch.nn.Sequential`等を使って複数のレイヤをまとめて定義する場合は、注意が必要である。  
その場合、個々のレイヤ名が自動で命名される。  
例えば、以下のように複数のレイヤをまとめて定義した場合、
  ```python
  import torch.nn as nn
  self.classifier = nn.Sequential(nn.Linear(out_ch_conv8 * 1 * 1, out_ch_fc1),  # 1
                                  nn.ReLU(True),  # model_infoへの記載対象外
                                  nn.Dropout(),  # model_infoへの記載対象外
                                  nn.Linear(out_ch_fc1, out_ch_fc2),  # 2
                                  nn.ReLU(True),  # model_infoへの記載対象外
                                  nn.Dropout(),  # model_infoへの記載対象外
                                  nn.Linear(out_ch_fc2, num_classes))  # 3(最終レイヤ)
  ```
  model_infoは、以下のように指定する。
  ```python
  model_info = OrderedDict()
  model_info['classifier.0'] = {'arg': 'out_ch_fc1'}  # 1
  model_info['classifier.3'] = {'arg': 'out_ch_fc2'}  # 2
  model_info['classifier.6'] = {'arg': None}  # 3(最終レイヤ)
  ```
* OrderedDictの各keyの値には、`{'arg': そのレイヤの出力引数名}`を指定する。  
ただし、最終レイヤの出力引数は`None`としておくこと。これは、最終レイヤの出力がpruningの対象外だからである。

#### 残差結合を含むネットワークの場合
対象レイヤの直前に残差結合が存在するときは、以下のようにその直前のレイヤ名を`'prev'`keyで指定すること。  
例として、サンプル`ResNet18`のネットワーク構成図の一部と、それに対するmodel_infoを示す。  

<img src="images/res18.png" width="900">

この場合、`'layer1.1.conv1'`レイヤと`'layer2.0.conv1'`レイヤの直前に残差結合(#1, #2)が存在する。
* 'layer1.1.conv1'レイヤの直前のレイヤ名は、'bn1'と'layer1.0.bn2'である。
* 'layer2.0.conv1'レイヤの直前のレイヤ名は、'bn1'と'layer1.0.bn2'と'layer1.1.bn2'である。  
これをmodel_infoに反映させると、以下のようになる。

```python
# Model information for pruning
model_info = OrderedDict()
(omitted)
model_info['layer1.1.conv1'] = {'arg': 'out_ch_l1_1_1', 'prev': ['bn1', 'layer1.0.bn2']}  # 1
(omitted)
model_info['layer2.0.conv1'] = {'arg': 'out_ch_l2_0_1', 'prev': ['bn1', 'layer1.0.bn2', 'layer1.1.bn2']}  # 2
(omitted)
```
詳細はサンプル`examples\ResNet18`を参照。  


### 3. auto-prune関数の実行

まず、`auto_prune.py`から`auto_prune`関数をimportする。  
import例は以下の通り。  
```python
from auto_prune import auto_prune
```

次に、ユーザーのモデルに合わせた引数を`auto_prune`関数に渡して実行する。  
実行例は以下の通り。
```python
weights, Afinal, n_args_channels = auto_prune(AlexNet, model_info, weights, Ab,
                                              train_loader, val_loader, criterion)
```
model_infoの指定方法は、`model_infoの設定`の項を参照。  
引数と返り値の詳細は`Docstring`の項を参照。  

#### Docstring

auto_pruning関数のDocstringを以下に示す。
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
`注) ResNet18などの残差結合を含むネットワークの場合、引数residual_connectionsをTrueに設定する必要がある。`

### 4. pruning後のモデルの使用方法

以下の例のように、pruning後のチャネル数を引数としてモデルをインスタンス化し、state_dictをロードする。
```python
from alexnet import AlexNet
model = AlexNet(**n_args_channels)  # n_args_channelsはauto_prune関数の返り値
model.load_state_dict(torch.load(pruned_model_path), strict=True)
```

## Tips

### 訓練済みモデルのkey名の書き換え方法

訓練済みモデルの重み等のkey名を変更する場合、モデルの重みをstate_dict()で一旦保存する。  
このとき、state_dictは単なるOrderedDict型であり、必要に応じてkeyの変更が可能である。  
モデルのkeyは`model.state_dict().keys()`で確認できる。  

### 特定のレイヤをpruning対象から除外

model_infoに、`'prune': False`を追加することで、特定のレイヤをpruning対象から除外することができる。  
例えば、AlexNetのmodel_infoで以下のように指定すると、`conv1, conv3, conv5, fc1`はpruning対象から除外される。
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

### モデルの圧縮率が低いとき
(江藤コメント: mattermostで相談させていただいた通り、まず私の思うところを書いていますので、ご確認ください⇒坂井さん)  

既定の設定でモデルの圧縮率が低い場合は、以下の対応を検討すること。

* epochsを増やす
* trust_radiusを増やす
* acc_marginを増やす(pruning後のAcc低下につながるため、要注意)

## Limitations  

* pruning対象とするレイヤは、torch.nn.Conv2d(or Conv1d))レイヤとtorch.nn.Linearレイヤのみ。  
ただし、BatchNorm2d(or BatchNorm1d)のnum_featuresも併せて削減する。

* 動作確認済みのネットワークは以下の通り
  * シリアルなネットワーク
    * AlexNet(examples/AlexNet)
    * BatchNormレイヤなしのVGG11(exmples/VGG11)
    * BatchNormレイヤありのVGG11(exmples/VGG11_bn)
    * 全結合レイヤのみで構成されるニューラルネット(examples/MLP)
  * 残差結合有りのネットワーク
    * ResNet18(examples/ResNet18)

* モデルの定義方法によっては、pruning実行にエラーする可能性あり。  

## Cautions

* 付属のサンプルは、本モジュールの効果を確認するためのものであり、実用には適さない場合がある。
* 本Readmeに記載しているコマンドは、実行環境に依存する。

## Copyright  

COPYRIGHT Fujitsu Limited 2021
