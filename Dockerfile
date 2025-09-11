FROM python:3.9-slim

WORKDIR /app

# システムの依存関係をインストール
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Pythonの依存関係をインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションファイルをコピー
COPY . .

# ポート設定
ENV PORT 8080
EXPOSE $PORT

# アプリケーション起動
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app