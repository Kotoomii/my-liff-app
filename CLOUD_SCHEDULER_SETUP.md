# Cloud Scheduler設定ガイド - モデル自動再学習

## 概要

毎朝、全ユーザーのMLモデルを自動的に再学習するためのCloud Schedulerの設定手順です。

## 前提条件

- Google Cloud Projectが作成済み
- Cloud Runにアプリケーションがデプロイ済み
- Cloud Scheduler APIが有効化されている

## 設定手順

### 1. Cloud Scheduler APIの有効化

```bash
gcloud services enable cloudscheduler.googleapis.com
```

### 2. Cloud Schedulerジョブの作成

#### GCPコンソールから設定する場合

1. **GCP Console** を開く
2. **Cloud Scheduler** にアクセス
3. **ジョブを作成** をクリック
4. 以下の情報を入力：

**基本設定:**
- **名前**: `model-retrain-daily`
- **リージョン**: `asia-northeast1` (東京)
- **説明**: `全ユーザーのMLモデルを毎日深夜0時に再学習`
- **頻度**: `0 0 * * *` (毎日深夜0時 JST)
- **タイムゾーン**: `Asia/Tokyo (JST)`

**実行内容の設定:**
- **ターゲットタイプ**: `HTTP`
- **URL**: `https://YOUR-CLOUD-RUN-URL/api/model/retrain-all`
  - 例: `https://my-liff-app-xxxxx-an.a.run.app/api/model/retrain-all`
- **HTTPメソッド**: `POST`
- **本文**: (空でOK)

**認証設定:**
- **Auth ヘッダー**: `Add OIDC token`
- **サービスアカウント**: Cloud Run起動元サービスアカウント
  - デフォルトの場合: `YOUR-PROJECT-ID@appspot.gserviceaccount.com`
- **Audience**: Cloud RunのURL (上記と同じ)

**再試行設定:**
- **最大再試行回数**: `3`
- **最大再試行期間**: `1h`
- **最小バックオフ**: `5s`
- **最大バックオフ**: `3600s`

5. **作成** をクリック

#### gcloudコマンドで設定する場合

```bash
# サービスアカウントの確認
gcloud projects describe YOUR-PROJECT-ID --format="value(projectNumber)"

# Cloud Schedulerジョブの作成
gcloud scheduler jobs create http model-retrain-daily \
  --location=asia-northeast1 \
  --schedule="0 0 * * *" \
  --time-zone="Asia/Tokyo" \
  --uri="https://YOUR-CLOUD-RUN-URL/api/model/retrain-all" \
  --http-method=POST \
  --oidc-service-account-email="YOUR-PROJECT-ID@appspot.gserviceaccount.com" \
  --oidc-token-audience="https://YOUR-CLOUD-RUN-URL" \
  --max-retry-attempts=3 \
  --max-retry-duration=1h \
  --description="全ユーザーのMLモデルを毎日深夜0時に再学習"
```

### 3. 動作確認

#### 手動でジョブを実行してテスト

```bash
gcloud scheduler jobs run model-retrain-daily --location=asia-northeast1
```

#### ログで結果を確認

```bash
# Cloud Runのログを確認
gcloud logging read "resource.type=cloud_run_revision AND textPayload:\"モデル再学習バッチ\"" \
  --limit=50 \
  --format=json
```

または、GCP Console > Cloud Run > ログ で確認

## スケジュール設定の説明

### Cron形式

`0 0 * * *` = 毎日0:00 JST（深夜0時）

- 第1項 (0): 分 (0-59)
- 第2項 (0): 時 (0-23)
- 第3項 (*): 日 (1-31)
- 第4項 (*): 月 (1-12)
- 第5項 (*): 曜日 (0-6, 0=日曜)

### スケジュール変更例

```bash
# 毎朝8時に変更
0 8 * * *

# 毎日正午に実行
0 12 * * *

# 毎週月曜日の深夜0時
0 0 * * 1

# 1日2回（深夜0時と正午）
0 0,12 * * *
```

## トラブルシューティング

### ジョブが実行されない場合

1. **Cloud Scheduler APIが有効か確認**
   ```bash
   gcloud services list --enabled | grep cloudscheduler
   ```

2. **サービスアカウントの権限確認**
   - Cloud Run起動元の権限が必要
   - IAM > サービスアカウントで確認

3. **ジョブの状態確認**
   ```bash
   gcloud scheduler jobs describe model-retrain-daily --location=asia-northeast1
   ```

### 認証エラーが出る場合

- OIDC トークンの設定を確認
- サービスアカウントがCloud Run Invokerロールを持っているか確認

### タイムアウトする場合

- Cloud Runのタイムアウト設定を延長（デフォルト300秒）
- ユーザー数が多い場合は実行時間が長くなる可能性あり

## APIエンドポイントの仕様

### リクエスト

```
POST /api/model/retrain-all
Content-Type: application/json
```

### レスポンス例

```json
{
  "status": "success",
  "timestamp": "2025-11-19T08:00:15.123456",
  "total_users": 10,
  "summary": {
    "success": 8,
    "error": 0,
    "skipped": 2
  },
  "users": [
    {
      "user_id": "default",
      "user_name": "デフォルトユーザー",
      "status": "success",
      "message": "モデル訓練完了",
      "data_count": 150,
      "metrics": {
        "rmse": 2.3456,
        "mae": 1.8765,
        "r2": 0.82
      }
    },
    {
      "user_id": "user1",
      "user_name": "小手川",
      "status": "insufficient_data",
      "message": "データ不足: 5件 < 10件",
      "data_count": 5
    }
  ]
}
```

## 参考リンク

- [Cloud Scheduler ドキュメント](https://cloud.google.com/scheduler/docs)
- [Cron形式リファレンス](https://cloud.google.com/scheduler/docs/configuring/cron-job-schedules)
