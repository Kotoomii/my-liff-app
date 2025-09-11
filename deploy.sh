#!/bin/bash

# GCP App Engine デプロイスクリプト
# フラストレーション値予測・反実仮想説明システム

set -e  # エラー時に停止

echo "🚀 GCP App Engine へのデプロイを開始します..."

# 必要な環境変数のチェック
if [ -z "$SPREADSHEET_ID" ]; then
    echo "⚠️  警告: SPREADSHEET_ID 環境変数が設定されていません"
    echo "   app.yaml の SPREADSHEET_ID を手動で設定してください"
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  警告: OPENAI_API_KEY 環境変数が設定されていません（LLMフィードバック機能は無効になります）"
fi

# サービスアカウントキーの存在確認
if [ ! -f "service-account-key.json" ]; then
    echo "❌ エラー: service-account-key.json が見つかりません"
    echo "   Google Cloud Console からサービスアカウントキーをダウンロードしてください"
    exit 1
fi

echo "📋 デプロイ前チェック..."

# Google Cloud SDK の確認
if ! command -v gcloud &> /dev/null; then
    echo "❌ エラー: Google Cloud SDK がインストールされていません"
    echo "   https://cloud.google.com/sdk/docs/install からインストールしてください"
    exit 1
fi

# 認証確認
echo "🔐 Google Cloud 認証状態を確認中..."
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 &> /dev/null; then
    echo "❌ エラー: Google Cloud にログインしていません"
    echo "   'gcloud auth login' を実行してください"
    exit 1
fi

# プロジェクト確認
PROJECT_ID=$(gcloud config get-value project)
if [ -z "$PROJECT_ID" ]; then
    echo "❌ エラー: Google Cloud プロジェクトが設定されていません"
    echo "   'gcloud config set project YOUR_PROJECT_ID' を実行してください"
    exit 1
fi

echo "✅ デプロイ準備完了"
echo "   プロジェクトID: $PROJECT_ID"
echo "   アカウント: $(gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1)"

# 必要なAPIの有効化
echo "🔧 必要なAPIを有効化中..."
gcloud services enable appengine.googleapis.com
gcloud services enable sheets.googleapis.com
gcloud services enable drive.googleapis.com

# app.yaml の環境変数を実際の値で更新（オプション）
if [ -n "$SPREADSHEET_ID" ]; then
    sed -i.bak "s/your-spreadsheet-id-here/$SPREADSHEET_ID/g" app.yaml
fi

if [ -n "$OPENAI_API_KEY" ]; then
    sed -i.bak "s/your-openai-api-key-here/$OPENAI_API_KEY/g" app.yaml
fi

# デプロイ実行
echo "🚀 App Engine へデプロイ中..."
gcloud app deploy app.yaml --quiet

# デプロイ後の情報表示
APP_URL=$(gcloud app browse --no-launch-browser)
echo "✅ デプロイ完了!"
echo ""
echo "🌐 アプリケーションURL: $APP_URL"
echo "📊 ヘルスチェック: $APP_URL/api/health"
echo "📈 フラストレーション予測API: $APP_URL/api/frustration/predict"
echo "🎯 DiCE分析API: $APP_URL/api/frustration/dice-analysis"
echo "💬 フィードバック生成API: $APP_URL/api/feedback/generate"
echo ""
echo "📝 ログの確認: gcloud app logs tail -s default"
echo "🔍 トラフィック確認: gcloud app versions list"

# バックアップファイルの削除
if [ -f "app.yaml.bak" ]; then
    rm app.yaml.bak
fi

echo "🎉 デプロイが正常に完了しました！"