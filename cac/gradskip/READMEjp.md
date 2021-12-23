# Gradient-Skip

Gradient-Skipは、学習を必要としていない層を見極めて計算をスキップさせる技術です。<br>
DLの誤差勾配から学習状況を取得して誤差勾配計算をスキップさせることで、誤差勾配算出分の計算量と通信量を削減します。<br>


## 学習スクリプトの修正
Gradient-Skipはpytorchのoptimizer側で実現される機能です。  
Gradient-Skipを実行するには、学習スクリプト側で利用するoptimizerをCAC_Libが提供するGradient-Skip版optimizerに差し替える必要があります。
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

## Gradient-Skipの実行
optimizerを差し替えたあとは、exampleの実行と同様の手順で実行することで、Gradient-skipを実行することができます。

### Gradient-Skipの動作方式
Gradient-Skipの動作方式にはマニュアルとオートの二通りの動作方式があります。

|動作方式|概要|
|:--|:--|
|マニュアル|停止させるレイヤー番号とイタレーション(※)を手動で指定する方式|
|オート|各レイヤーの種類と重み分散に基づき、停止させるレイヤーとタイミングが自動的に決定される方式|

(※)ここで言うイタレーションとはoptimizerの呼び出し回数のことを指します。学習スクリプトにおける学習処理の繰り返し回数ではないことに注意してください。

Gradient-Skipの初期状態では、デフォルト(推奨)の閾値にてオート方式が動作するようになっています。<br>
オート方式の閾値を変更したり、マニュアル方式で動作させたい場合は、次節に示す環境変数を設定してください。

### Gradient-Skip用環境変数の設定
Gradient-Skipでは下表に示す環境変数が用意されています。マニュアル方式・オート方式それぞれに固有の環境変数と、共通の環境変数があります。<br>
`CAC_STOP_LAYER_NUM`と`CAC_STOP_LAYER_ITR`が設定されている場合はマニュアル方式となります。オート方式の環境変数が同時に設定されている場合はマニュアル方式が優先されます。

|環境変数|動作方式|デフォルト値|機能|
|:--|:--|:-:|:--|
|CAC_STOP_LAYER_NUM|マニュアル|なし|レイヤー全体の動作を停止させるレイヤーを指定する。|
|CAC_STOP_LAYER_ITR|マニュアル|なし|`CAC_STOP_LAYER_NUM`で指定されるレイヤー全体の動作の停止を有効化するイタレーションを指定する。|
|CAC_VAR_START_ITR|オート|5000|レイヤー停止判断を開始するイタレーションを指定する。|
|CAC_VAR_START_THR|オート|0.95|最初の停止対象レイヤーにおいて、レイヤーを停止する基準となる重み分散の閾値を指定する。|
|CAC_VAR_MT_COUNT_THR|オート|5|最初の停止対象レイヤーを除く各停止対象レイヤーにおいて、そのレイヤーの重み分散が山型に変化していると判断する基準となる、重み分散ピークの更新回数を指定する。|
|CAC_VAR_MT_THR|オート|0.96|重み分散が山型に変化している停止対象レイヤーにおいて、レイヤーを停止する基準となる重み分散の閾値を指定する。|
|CAC_VAR_SLOPE_THR|オート|0.98|重み分散がスロープ型に変化している停止対象レイヤーにおいて、レイヤーを停止する基準となる重み分散の閾値を指定する。|
|CAC_VAR_SAMPLES|オート|200|レイヤー停止判断を行うイタレーションの間隔を指定する。|
|CAC_BRAKING_DISTANCE|共通|0|Braking Distanceを利用する場合、Braking Distanceで学習率を下げる期間（イタレーション）を指定する。|

pytorchの実行時、必要に応じてこれらの環境変数を設定してください。
あらかじめコマンドラインから設定しておくか、pytorchの実行スクリプトに設定を記述しておきます（実行スクリプトへの記述例は、CAC_Libのexampleに含まれる`example_imagenet.sh`を参照）。
#### マニュアル方式での環境変数設定例
```bash
export CAC_STOP_LAYER_NUM=10,20
export CAC_STOP_LAYER_ITR=0,100
```
#### オート方式での環境変数設定例
```bash
export CAC_VAR_START_ITR=5000
export CAC_VAR_START_THR=0.95
export CAC_VAR_MT_COUNT_THR=5
export CAC_VAR_MT_THR=0.96
export CAC_VAR_SLOPE_THR=0.98
export CAC_VAR_SAMPLES=200
export CAC_BRAKING_DISTANCE=300  # Braking Distance利用時のみ設定
```
