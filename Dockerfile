# ベースイメージはPyTorch（CUDA環境を含む）を利用
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# 作業ディレクトリの設定
WORKDIR /app

# OpenCV の動作に必要なシステムライブラリ（libgl1など）をインストール
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# まず requirements.txt をコピーしてPython依存関係をインストール（キャッシュ効果を高めるため）
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# コード全体をコンテナにコピー
COPY . /app/

# デフォルトのコマンド (推論スクリプト)
# docker run を行う際に --dataset などを引数として渡せます
ENTRYPOINT ["python", "run_inference.py"]
