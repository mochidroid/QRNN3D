# QRNN3D 推論スクリプトの使い方

本リポジトリでは、学習済みのモデルを用いてハイパースペクトル画像のノイズ除去（推論）を行うための専用スクリプト `run_inference.py` と、実行に必要な専用のPython環境を構築しました。

## 1. 構築した内容

- **推論スクリプト (`run_inference.py`)**: 
  - ノイズが付加されたハイパースペクトル画像を受け取り、モデルによるノイズ除去を行います。
  - MPSNR, MSSIM, SAM などの定量的な指標を計算し、ターミナルに出力します。
  - 指定された特定の波長（バンド）の画像を PNG 形式で保存し、視覚的な確認ができるようにしています。
  - 結果のテンソルを `.mat` 形式で適切に保存します。
- **指標計算処理の修正 (`utility/indexes.py`)**: 
  - 最新の `scikit-image` (skimage.metrics) に対応するように SSIM / PSNR の計算関数を修正しました。
- **専用 Python 環境 (`qrnn3d_env`)**:
  - `torchnet` などの競合エラーやバージョンの問題を避けるため、QRNN3D を動かすために必要最小限かつ最新の環境（PyTorch, SciPy, OpenCV など）を分離して構築しました。

## 2. 実行環境の準備

スクリプトを実行する前に、今回作成した専用の仮想環境を有効化する必要があります。ターミナルで以下のコマンドを実行してください。

```bash
cd /root/repos/QRNN3D
source qrnn3d_env/bin/activate
```

※ もしくは環境を activate せずに、直接 `qrnn3d_env/bin/python` を使用して実行することも可能です。

## 3. 基本的な実行方法

特に引数を指定しない場合、デフォルトで以下の設定が使用されます。
- 入力パス: `dataset/normalized/JasperRidge/Case1/data.mat`
- 出力ディレクトリ: `result/normalized/JasperRidge/Case1`
- モデルパス: `checkpoints/qrnn3d/complex/model_epoch_100_159904.pth`
- 保存する画像バンド: `50`

**実行コマンド:**
```bash
python run_inference.py
```

実行が完了すると、`result/normalized/JasperRidge/Case1/` ディレクトリに以下のファイルが生成されます。
- `restored.mat`: ノイズ除去された結果 (restored), 正解データ (gt), 元のノイズ画像 (input) が含まれた MATLAB ファイル
- `gt_band50.png`: 正解データの画像
- `input_band50.png`: ノイズが付加された入力画像
- `restored_band50.png`: ノイズ除去後の画像

## 4. カスタム条件での実行方法 (CLI 引数)

データセット名や出力するバンドなどを変えたい場合は、コマンドライン引数で自由に指定することができます。

**使用できる引数:**
- `--input_path` : 入力データ (`.mat` ファイル) のパス（例: `dataset/normalized/PaviaU/Case2/data.mat` など）
- `--output_dir` : 推論結果を保存するディレクトリのパス（例: `result/normalized/PaviaU/Case2` など）
- `--checkpoint` : 使用する PyTorch のチェックポイントモデルのパス
- `--band` : プレビュー画像として抽出・表示したいバンド（波長）のインデックス（※ 0始まり）

**実行例 1: PaviaU データセットの Case2 に対して実行し、バンド 30 を出力する**
```bash
python run_inference.py \
    --input_path dataset/normalized/PaviaU/Case2/data.mat \
    --output_dir result/normalized/PaviaU/Case2 \
    --checkpoint checkpoints/qrnn3d/complex/model_epoch_100_159904.pth \
    --band 30
```

**実行例 2: 別のチェックポイントファイルを使用して実行する**
```bash
python run_inference.py \
    --checkpoint checkpoints/qrnn3d/gauss/model_epoch_50_118454.pth \
    --band 100
```
