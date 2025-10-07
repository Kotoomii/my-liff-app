# Cloud Runログ最適化ガイド

このドキュメントでは、GCP Cloud Run上でのログコスト削減のための設定方法を説明します。

## 問題

Cloud Runのオブザーバビリティ機能では、ログの量に応じて課金が発生します。
デフォルト設定では大量のINFO/DEBUGログが出力され、課金が増大する可能性があります。

## 解決策

### 1. ログレベルの最適化

本番環境では`LOG_LEVEL=WARNING`に設定することで、重要なログ（WARNING以上）のみを出力します。

```bash
# Cloud Run環境変数設定例
LOG_LEVEL=WARNING
```

### 2. 詳細ログの制御

機能別にログ出力を制御できるフラグを追加しました：

| 環境変数 | 用途 | 本番推奨値 | 開発推奨値 |
|---------|------|-----------|-----------|
| `ENABLE_DEBUG_LOGS` | デバッグログ | `false` | `true` |
| `ENABLE_INFO_LOGS` | 情報ログ | `false` | `true` |
| `LOG_PREDICTIONS` | 予測結果ログ | `false` | `true` |
| `LOG_DATA_OPERATIONS` | データ操作ログ | `false` | `true` |
| `LOG_MODEL_TRAINING` | モデル訓練ログ | `false` | `true` |

### 3. 構造化ログ

Cloud Run環境では自動的にJSON形式の構造化ログを出力します。
これにより、Cloud Loggingでの検索・フィルタリングが容易になります。

```json
{
  "severity": "WARNING",
  "message": "予測値のバラつきが小さいです",
  "timestamp": "2025-01-15T10:30:00Z",
  "user_id": "default"
}
```

## Cloud Run環境変数設定方法

### gcloud CLIを使用

```bash
gcloud run services update YOUR_SERVICE_NAME \
  --region YOUR_REGION \
  --update-env-vars LOG_LEVEL=WARNING,ENABLE_INFO_LOGS=false,LOG_PREDICTIONS=false,LOG_DATA_OPERATIONS=false,LOG_MODEL_TRAINING=false
```

### Cloud Consoleを使用

1. Cloud Run サービスページを開く
2. サービスを選択
3. 「新しいリビジョンの編集とデプロイ」をクリック
4. 「コンテナ」タブ → 「変数とシークレット」
5. 以下の環境変数を追加：
   - `LOG_LEVEL` = `WARNING`
   - `ENABLE_INFO_LOGS` = `false`
   - `LOG_PREDICTIONS` = `false`
   - `LOG_DATA_OPERATIONS` = `false`
   - `LOG_MODEL_TRAINING` = `false`

## ログレベル別の出力内容

### WARNING（本番推奨）
- データ品質警告
- 予測精度低下の警告
- 重要なエラー情報

### INFO（開発環境）
- アプリケーション起動/終了
- スケジューラー実行
- 予測結果
- データ操作

### DEBUG（デバッグ時のみ）
- 詳細な処理フロー
- 中間データ
- タイムライン詳細

## 期待される効果

本番環境で推奨設定を適用することで：

1. **ログ量の削減**: 約80-90%のログ削減
2. **コスト削減**: Cloud Loggingの課金を大幅に削減
3. **重要情報の明確化**: WARNINGレベル以上のみ出力されるため、問題の早期発見が容易

## トラブルシューティング時の設定

問題調査時には一時的にログレベルを上げることができます：

```bash
# 一時的にINFOログを有効化
gcloud run services update YOUR_SERVICE_NAME \
  --update-env-vars LOG_LEVEL=INFO,ENABLE_INFO_LOGS=true
```

調査完了後は必ず元の設定に戻してください。

## ローカル開発環境

ローカルでは詳細ログを有効にして開発効率を上げましょう：

```bash
# .env ファイル
LOG_LEVEL=INFO
ENABLE_DEBUG_LOGS=true
ENABLE_INFO_LOGS=true
LOG_PREDICTIONS=true
LOG_DATA_OPERATIONS=true
LOG_MODEL_TRAINING=true
```

## 参考リンク

- [Cloud Logging 料金](https://cloud.google.com/logging/pricing)
- [Cloud Run ログ](https://cloud.google.com/run/docs/logging)
