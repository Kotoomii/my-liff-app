# Cloud Scheduler セットアップガイド

このドキュメントでは、`min_instances=1` から `min_instances=0 + Cloud Scheduler` への移行手順と、必要に応じてロールバックする方法を説明します。

## コスト削減効果

- **現状 (min_instances=1)**: 約 $65-70/月
- **Cloud Scheduler使用**: 約 $5-10/月
- **削減額**: 約 **85%のコスト削減**

---

## 📋 前提条件

- Google Cloud プロジェクトが作成されている
- Cloud Run サービスがデプロイ済み
- `gcloud` コマンドがインストールされている

---

## 🚀 移行手順

### ステップ 1: 認証トークンの設定

Cloud Scheduler からのリクエストを認証するため、環境変数にトークンを設定します。

```bash
# ランダムな認証トークンを生成
SCHEDULER_TOKEN=$(openssl rand -hex 32)
echo "生成されたトークン: $SCHEDULER_TOKEN"

# Cloud Run サービスに環境変数を設定
gcloud run services update research-matsui \
  --set-env-vars SCHEDULER_AUTH_TOKEN=$SCHEDULER_TOKEN \
  --region asia-northeast1
```

**⚠️ 重要**: 生成されたトークンは安全な場所に保存してください（次のステップで使用します）。

---

### ステップ 2: Cloud Scheduler ジョブの作成

#### 2-1. データ監視ジョブ（9:00-22:00、毎時0分・30分）

```bash
# サービスURLを取得
SERVICE_URL=$(gcloud run services describe research-matsui --region asia-northeast1 --format='value(status.url)')

# Cloud Scheduler ジョブを作成（0分実行）
gcloud scheduler jobs create http data-monitor-00 \
  --location asia-northeast1 \
  --schedule "0 0-13 * * *" \
  --time-zone "Asia/Tokyo" \
  --uri "${SERVICE_URL}/api/scheduler/monitor" \
  --http-method POST \
  --headers "X-Scheduler-Auth=${SCHEDULER_TOKEN}" \
  --attempt-deadline 600s \
  --description "データ監視（9:00-22:00、毎時0分）"

# Cloud Scheduler ジョブを作成（30分実行）
gcloud scheduler jobs create http data-monitor-30 \
  --location asia-northeast1 \
  --schedule "30 0-13 * * *" \
  --time-zone "Asia/Tokyo" \
  --uri "${SERVICE_URL}/api/scheduler/monitor" \
  --http-method POST \
  --headers "X-Scheduler-Auth=${SCHEDULER_TOKEN}" \
  --attempt-deadline 600s \
  --description "データ監視（9:00-22:00、毎時30分）"
```

**スケジュール説明**:
- `0 0-13 * * *`: 毎日 0:00-13:00（UTC）= 9:00-22:00（JST）の毎時0分
- `30 0-13 * * *`: 毎日 0:30-13:30（UTC）= 9:30-22:30（JST）の毎時30分

#### 2-2. DiCE実行ジョブ（22:10 JST）

```bash
# DiCE実行ジョブを作成
gcloud scheduler jobs create http dice-evening \
  --location asia-northeast1 \
  --schedule "10 13 * * *" \
  --time-zone "Asia/Tokyo" \
  --uri "${SERVICE_URL}/api/scheduler/dice" \
  --http-method POST \
  --headers "X-Scheduler-Auth=${SCHEDULER_TOKEN}" \
  --attempt-deadline 1800s \
  --description "DiCE実行 + フィードバック生成（22:10 JST）"
```

**スケジュール説明**:
- `10 13 * * *`: 毎日 13:10（UTC）= 22:10（JST）

---

### ステップ 3: min_instances を 0 に変更

```bash
# Cloud Run サービスの min_instances を 0 に変更
gcloud run services update research-matsui \
  --min-instances 0 \
  --region asia-northeast1
```

これで、Cloud Scheduler による起動のみが有効になります。

---

## ✅ 動作確認

### 手動でジョブをトリガー

```bash
# データ監視ジョブを手動実行
gcloud scheduler jobs run data-monitor-00 --location asia-northeast1

# DiCEジョブを手動実行
gcloud scheduler jobs run dice-evening --location asia-northeast1
```

### ログの確認

```bash
# Cloud Run ログを確認
gcloud run logs read research-matsui --region asia-northeast1 --limit 50
```

---

## 🔄 ロールバック手順（問題が発生した場合）

Cloud Scheduler に問題が発生した場合、すぐに元の `min_instances=1` に戻すことができます。

### ステップ 1: min_instances を 1 に戻す

```bash
# すぐに元に戻す
gcloud run services update research-matsui \
  --min-instances 1 \
  --region asia-northeast1
```

これで、既存の `data_monitor_loop()` が自動的に起動します。

### ステップ 2: Cloud Scheduler ジョブの一時停止（オプション）

```bash
# ジョブを一時停止（削除はしない）
gcloud scheduler jobs pause data-monitor-00 --location asia-northeast1
gcloud scheduler jobs pause data-monitor-30 --location asia-northeast1
gcloud scheduler jobs pause dice-evening --location asia-northeast1
```

---

## 📊 スケジュール実行頻度の比較

| 項目 | 現状（min_instances=1） | Cloud Scheduler |
|------|------------------------|----------------|
| データ監視 | 9:00-22:00、毎時 0,15,30,45分 (52回/日) | 9:00-22:00、毎時 0,30分 (27回/日) |
| DiCE実行 | 22:10 JST (1回/日) | 22:10 JST (1回/日) |
| 合計実行回数 | 53回/日 | 28回/日 |
| コスト | $65-70/月 | $5-10/月 |

**削減率**: 約 **85%のコスト削減**

---

## 🔍 トラブルシューティング

### エラー: 認証失敗

```
⚠️ Cloud Scheduler認証失敗
```

**解決方法**:
1. Cloud Run サービスの環境変数 `SCHEDULER_AUTH_TOKEN` を確認
2. Cloud Scheduler ジョブのヘッダー `X-Scheduler-Auth` を確認
3. トークンが一致しているか確認

### エラー: タイムアウト

```
Cloud Scheduler DiCE実行エラー
```

**解決方法**:
1. `--attempt-deadline` を増やす（例: 1800s → 3600s）
2. ログを確認してボトルネックを特定

```bash
gcloud run logs read research-matsui --region asia-northeast1 --limit 100
```

---

## 📝 注意事項

1. **既存コードはそのまま残しています**
   - `data_monitor_loop()` と `scheduler._execute_evening_feedback()` は削除していません
   - `min_instances=1` に戻すだけで、すぐに元の動作に戻ります

2. **新しいエンドポイント**
   - `/api/scheduler/monitor`: Cloud Scheduler 用データ監視エンドポイント
   - `/api/scheduler/dice`: Cloud Scheduler 用 DiCE 実行エンドポイント

3. **セキュリティ**
   - 認証トークンは環境変数で管理
   - より強固なセキュリティが必要な場合は、Cloud Scheduler の OIDC トークン認証を推奨

---

## 📞 サポート

問題が発生した場合は、すぐに `min_instances=1` に戻してください。

```bash
gcloud run services update research-matsui \
  --min-instances 1 \
  --region asia-northeast1
```
