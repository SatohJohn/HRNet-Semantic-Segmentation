## 概要

[こちら](https://github.com/HRNet/HRNet-Semantic-Segmentation)の結果画像を出力がしたいがためにフォークをしたもの。
gpu環境でのものではないので注意する。

## 共通

### 準備

1. テスト画像を data/dataset/ に配置する
2. data/list/test.lst というファイルを作成し、テスト画像のパスを以下の例のように記載する
```
dataset/testing_images/100012_501646.jpg
dataset/testing_images/100095_453067.jpg
dataset/testing_images/100098_193288.jpg
```
3. https://github.com/hsfzxjy/models.storage/releases/download/HRNet-OCR/hrnet_ocr_lip_5648_torch04.pth から学習済みmodelをdownloadする

## lip

### 準備

1. `docker build -t hrnet:lip -f lip/Dockerfile .` を実行してDockerImageを作成する
2. `docker run -v `\`pwd\``:/app  -it hrnet:lip` でプロジェクトをmountして、1で作成したDockerImageを起動する

### 実行

```
python tools/test.py \
--cfg experiments/lip.yaml DATASET.TEST_SET list/test.lst \
TEST.MODEL_FILE {共通の準備3でdownloadしたモデルのパス} TEST.FLIP_TEST True \
TEST.NUM_SAMPLES 0 TEST.MULTI_SCALE True \
MODEL.PRETRAINED {共通の準備3でdownloadしたモデルのパス}
```
