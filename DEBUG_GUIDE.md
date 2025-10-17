
# 機械学習モデル デバッグガイド

このガイドでは、予測値が固定される問題をデバッグする方法を説明します。

## デバッグツール

### 1. デバッグAPIエンドポイント

#### `/api/debug/model`

モデルの詳細な診断情報を取得します。

```bash
curl "https://your-app.run.app/api/debug/model?user_id=default"
```

**レスポンス内容:**
- `data_quality`: データ品質評価
- `nasa_f_stats`: NASA_F値の統計（平均、標準偏差、分布）
- `activity_stats`: 活動カテゴリの統計
- `model_info`: モデルの訓練状態、特徴量数
- `encoders_info`: エンコーダーの状態
- `wfv_results`: Walk Forward Validation結果、特徴量重要度
- `test_predictions`: テスト予測（固定値 vs 履歴使用）
- `issues`: 検出された問題のリスト
- `diagnosis`: 総合診断結果

### 2. ローカルデバッグスクリプト

```bash
cd /path/to/research_Matsui
python debug_ml_model.py
```

**出力内容:**
- データ統計
- NASA_F分布
- 活動カテゴリ分析
- モデル訓練結果
- 特徴量重要度
- テスト予測
- 診断結果

## よくある問題と解決方法

### 問題1: 予測値が固定される（例: 8.72, 10.7）

**原因チェックリスト:**

#### ✅ 未来データの混入
```bash
# デバッグAPIで確認
curl "https://your-app.run.app/api/debug/model?user_id=default" | jq '.test_predictions'
```

- `historical_records` が各予測でほぼ同じ → 未来データが混入している可能性
- **修正済み**: `predict_with_history()` で現在時刻より前のデータのみ使用

#### ✅ 活動カテゴリがNone

**ログ例:**
```
⚠️ 予測に固定値の特徴量を使用しています。活動: None
```

**原因:** `CatSub` カラムがデータに存在しないか空

**修正済み:**
```python
activity_category = data.get('CatSub', 'その他')  # デフォルト値を設定
if not activity_category or activity_category.strip() == '':
    activity_category = 'その他'
```

#### ✅ NASA_Fの分散が小さい

```bash
# 確認
curl "https://your-app.run.app/api/debug/model?user_id=default" | jq '.nasa_f_stats'
```

```json
{
  "std": 0.85,  // 1.0未満なら問題
  "unique_values": 2  // 3未満なら問題
}
```

**解決策:**
- 異なるフラストレーション値を記録
- 低ストレス活動（5-8）、中ストレス活動（9-12）、高ストレス活動（13-18）

#### ✅ データ数不足

```json
{
  "data_quality": {
    "total_samples": 8,  // 10未満
    "is_sufficient": false
  }
}
```

**解決策:**
- 最低10件以上のデータを蓄積
- 推奨30件以上

#### ✅ 活動の多様性不足

```json
{
  "activity_stats": {
    "unique": 2  // 3未満なら問題
  }
}
```

**解決策:**
- 様々な種類の活動を記録

### 問題2: 予測値の多様性が低い

**ログ例:**
```
⚠️ 警告: 予測値の多様性が低い（標準偏差: 0.000）
```

**確認:**
```bash
curl "https://your-app.run.app/api/debug/model?user_id=default" | \
  jq '.wfv_results.prediction_diversity'
```

```json
{
  "std": 0.15,  // 0.5未満なら問題
  "unique_values": 2,  // 3未満なら問題
  "is_diverse": false
}
```

**原因:**
1. データの分散不足
2. 特徴量の多様性不足
3. 履歴データに同じ値が含まれている

**解決策:**
- NASA_F値に変化をつける
- 異なる時間帯・曜日のデータを記録
- 活動の種類を増やす

### 問題3: モデルが訓練されていない

```json
{
  "model_info": {
    "is_trained": false
  }
}
```

**原因:**
- データ数が10件未満
- 初回アクセス

**解決策:**
- データを10件以上蓄積
- タイムラインAPIアクセス時に自動訓練

## デバッグ手順

### Step 1: 現状把握

```bash
# デバッグAPIで全体状況を確認
curl "https://your-app.run.app/api/debug/model?user_id=default" \
  -o debug_result.json

# 問題点を確認
cat debug_result.json | jq '.issues'
```

### Step 2: データ統計確認

```bash
# NASA_F統計
cat debug_result.json | jq '.nasa_f_stats'

# 活動統計
cat debug_result.json | jq '.activity_stats'

# データ品質
cat debug_result.json | jq '.data_quality'
```

### Step 3: モデル状態確認

```bash
# モデル情報
cat debug_result.json | jq '.model_info'

# 特徴量重要度
cat debug_result.json | jq '.wfv_results.feature_importance'

# 予測多様性
cat debug_result.json | jq '.wfv_results.prediction_diversity'
```

### Step 4: テスト予測確認

```bash
# 固定値予測 vs 履歴使用予測の比較
cat debug_result.json | jq '.test_predictions'
```

**期待される結果:**
```json
[
  {
    "activity": "仕事",
    "actual": 12.0,
    "predicted_fixed": 10.7,  // 固定値（いつも同じ）
    "predicted_history": 11.5,  // 履歴使用（多様）
    "historical_records": 15
  }
]
```

### Step 5: Cloud Runログ確認

```bash
# エラーログのみ表示
gcloud logging read "resource.type=cloud_run_revision AND severity>=WARNING" \
  --limit 50 --format json
```

**注目すべきログ:**
- `⚠️ 予測に固定値の特徴量を使用`
- `⚠️ 予測値の多様性が低い`
- `⚠️ データ品質警告`

## 環境変数設定（デバッグ時）

Cloud Runでデバッグログを有効化:

```bash
gcloud run services update YOUR_SERVICE \
  --update-env-vars \
    LOG_LEVEL=INFO,\
    ENABLE_DEBUG_LOGS=true,\
    LOG_PREDICTIONS=true,\
    LOG_MODEL_TRAINING=true
```

## トラブルシューティング例

### ケース1: すべて10.7で固定

**診断:**
```bash
curl "https://your-app.run.app/api/debug/model?user_id=default" | jq '.test_predictions'
```

**結果:**
```json
"predicted_fixed": 10.7,
"predicted_history": 10.7,
"historical_records": 50
```

→ 履歴使用でも同じ = データの問題

**確認:**
```bash
curl "https://your-app.run.app/api/debug/model?user_id=default" | jq '.nasa_f_stats'
```

→ `std: 0.5` → データの分散不足

**解決:** NASA_F値に変化をつけて記録

### ケース2: 活動がNone

**ログ:**
```
⚠️ 予測に固定値の特徴量を使用しています。活動: None
```

**修正済み:** デフォルト値「その他」を設定

### ケース3: 履歴データが使われていない

**診断:**
```bash
curl "https://your-app.run.app/api/debug/model?user_id=default" | jq '.test_predictions'
```

**結果:**
```json
"historical_records": 0
```

→ 履歴データが取得できていない

**確認ポイント:**
- `Timestamp` カラムが存在するか
- `NASA_F` カラムが存在するか
- 現在時刻より前のデータがあるか

## まとめ

### チェックリスト

- [ ] データ数: 10件以上
- [ ] NASA_F標準偏差: 1.0以上
- [ ] NASA_Fユニーク値: 3種類以上
- [ ] 活動種類: 3種類以上
- [ ] モデル訓練済み
- [ ] 予測値多様性: 標準偏差0.5以上
- [ ] 活動カテゴリ: None禁止
- [ ] 履歴データ使用: historical_records > 0

すべてクリアすれば、多様な予測値が得られます。
