# ユーザーごとのモデル管理実装

## 概要

「モデルが学習されていません」エラーを解決し、ユーザーごとに独立した機械学習モデルを管理する機能を実装しました。

## 実装内容

### 1. ユーザーごとのモデル管理

```python
# グローバル変数でユーザーごとのpredictorインスタンスを管理
user_predictors = {}  # {user_id: FrustrationPredictor}

def get_predictor(user_id: str) -> FrustrationPredictor:
    """
    ユーザーごとのpredictorを取得（存在しない場合は作成）
    """
    if user_id not in user_predictors:
        logger.info(f"新しいpredictorを作成: user_id={user_id}")
        user_predictors[user_id] = FrustrationPredictor()
    return user_predictors[user_id]
```

**利点:**
- ユーザーごとに独立したモデル、エンコーダー、特徴量を管理
- ユーザーAのデータがユーザーBのモデルに影響しない
- ユーザーごとに異なる活動カテゴリや生活パターンに対応

### 2. 自動モデル訓練

```python
def ensure_model_trained(user_id: str, force_retrain: bool = False) -> dict:
    """
    ユーザーのモデルが訓練されていることを確認
    訓練されていない場合は自動的に訓練を実行
    """
    predictor = get_predictor(user_id)

    # すでに訓練済みで強制再訓練でない場合はスキップ
    if predictor.model is not None and not force_retrain:
        return {
            'status': 'already_trained',
            'message': 'モデルはすでに訓練されています'
        }

    # データ取得
    activity_data = sheets_connector.get_activity_data(user_id)
    fitbit_data = sheets_connector.get_fitbit_data(user_id)

    if activity_data.empty:
        return {
            'status': 'no_data',
            'message': 'スプレッドシートにデータがありません'
        }

    # データ前処理
    activity_processed = predictor.preprocess_activity_data(activity_data)
    df_enhanced = predictor.aggregate_fitbit_by_activity(activity_processed, fitbit_data)

    # データ品質チェック
    data_quality = predictor.check_data_quality(df_enhanced)

    if not data_quality['is_sufficient']:
        return {
            'status': 'insufficient_data',
            'message': f"データ数が不足しています（{len(df_enhanced)}件/最低10件必要）",
            'data_quality': data_quality
        }

    # Walk Forward Validationで訓練
    try:
        training_results = predictor.walk_forward_validation_train(df_enhanced)
        return {
            'status': 'success',
            'message': f'モデルの訓練が完了しました（{len(df_enhanced)}件のデータ）',
            'training_results': training_results
        }
    except Exception as e:
        logger.error(f"モデル訓練エラー: {e}")
        return {
            'status': 'training_error',
            'message': f'モデルの訓練中にエラーが発生しました: {str(e)}'
        }
```

**動作:**
1. ユーザーのモデルが訓練済みかチェック
2. 未訓練の場合、スプレッドシートからデータを取得
3. データ品質を確認（最低10件必要）
4. Walk Forward Validationで自動訓練
5. 訓練結果を返す

### 3. 各APIエンドポイントの対応

#### タイムラインAPI (`/api/data/timeline`)

```python
# ユーザーごとのpredictorを取得
predictor = get_predictor(user_id)

# モデルが訓練されていない場合は自動訓練
training_info = {'auto_trained': False}
if predictor.model is None:
    logger.info(f"モデル未訓練: user_id={user_id}, 自動訓練を開始します")
    training_result = ensure_model_trained(user_id)
    training_info = {
        'auto_trained': True,
        'status': training_result.get('status'),
        'message': training_result.get('message')
    }

# レスポンスにtraining_infoを追加
return jsonify({
    'status': 'success',
    'user_id': user_id,
    'timeline': timeline,
    'training_info': training_info  # 訓練情報を含める
})
```

**レスポンス例（初回訓練時）:**
```json
{
  "status": "success",
  "user_id": "default",
  "timeline": [...],
  "training_info": {
    "auto_trained": true,
    "status": "success",
    "message": "モデルの訓練が完了しました（50件のデータ）"
  }
}
```

**レスポンス例（訓練済み）:**
```json
{
  "status": "success",
  "user_id": "default",
  "timeline": [...],
  "training_info": {
    "auto_trained": false
  }
}
```

#### 予測API (`/api/frustration/predict`)

```python
# ユーザーごとのpredictorを取得
predictor = get_predictor(user_id)

# モデルが訓練されていない場合は自動訓練
if predictor.model is None:
    training_result = ensure_model_trained(user_id)
    # ...
```

#### 活動予測API (`/api/frustration/predict-activity`)

```python
# ユーザーごとのpredictorを取得
predictor = get_predictor(user_id)

# モデルが訓練されていない場合は自動訓練
if predictor.model is None and len(df_enhanced) >= 10:
    ensure_model_trained(user_id)
```

#### デバッグAPI (`/api/debug/model`)

```python
# ユーザーごとのpredictorを取得
predictor = get_predictor(user_id)

# ユーザー固有のモデル状態を確認
model_info = {
    'is_trained': predictor.model is not None,
    'feature_count': len(predictor.feature_columns) if predictor.feature_columns else 0,
    # ...
}
```

## 使用方法

### 新規ユーザーの場合

1. タブレットアプリでデータを記録（活動、フラストレーション値）
2. データが10件以上蓄積されたら自動的にモデル訓練が開始
3. タイムラインAPIにアクセスした際に自動訓練される

### 既存ユーザーの場合

- 一度モデルが訓練されると、以降はそのモデルを使用して予測
- Cloud Runコンテナが再起動するまで、メモリ上にモデルを保持

### 強制再訓練

```python
# ensure_model_trained()にforce_retrain=Trueを渡す
training_result = ensure_model_trained(user_id, force_retrain=True)
```

## データフロー

```
1. ユーザーがタイムラインAPIにアクセス
   ↓
2. get_predictor(user_id) でユーザー固有のpredictorを取得
   ↓
3. モデルが未訓練？
   Yes → ensure_model_trained(user_id)
      ↓
      a. スプレッドシートからデータ取得
      ↓
      b. データ品質チェック（10件以上？）
      ↓
      c. Walk Forward Validationで訓練
      ↓
      d. モデルをpredictorに保存
   No → そのまま予測処理へ
   ↓
4. 予測実行
   ↓
5. レスポンス返却（training_info含む）
```

## 注意事項

### 1. モデルの永続性

**現状の制限:**
- モデルはメモリ上（`user_predictors` dict）に保存
- Cloud Runコンテナが再起動すると消失
- 次回アクセス時に再訓練が必要

**影響:**
- コールドスタート時に初回アクセスが遅くなる可能性
- ただし、データがあれば自動的に再訓練されるため機能は維持

**将来の改善案:**
- Cloud Storageにモデルを保存
- Pickleファイルとしてスプレッドシートに保存
- モデルバージョニング管理

### 2. メモリ使用量

- ユーザー数が増えると、`user_predictors`のメモリ使用量が増加
- Cloud Runのメモリ制限（512MB〜4GB）に注意
- 必要に応じてLRU（Least Recently Used）キャッシュの実装を検討

### 3. 並行アクセス

- 複数リクエストが同時にモデル訓練を開始する可能性
- 現状は最後の訓練結果が保存される（上書き）
- 必要に応じてロック機構の実装を検討

## トラブルシューティング

### 「モデルが学習されていません」エラー

**原因:**
- データが10件未満
- スプレッドシートが空

**解決:**
```bash
# デバッグAPIで確認
curl "https://your-app.run.app/api/debug/model?user_id=default"
```

確認ポイント:
- `data_quality.total_samples` が10以上か
- `model_info.is_trained` がtrueか

### モデル訓練が自動で開始されない

**確認:**
1. タイムラインAPIのレスポンスで`training_info`を確認
2. Cloud Runログで"モデル未訓練"メッセージを確認

**原因:**
- データ数不足
- データ取得エラー

### 予測値が固定される

**原因:**
- ユーザー固有のモデルではなく、古いグローバルpredictorを使用している可能性

**確認:**
```python
# デバッグAPIでエンコーダーやモデル状態を確認
curl "https://your-app.run.app/api/debug/model?user_id=default"
```

## まとめ

この実装により:

✅ ユーザーごとに独立したモデルを管理
✅ スプレッドシートからデータを自動取得して訓練
✅ 「モデルが学習されていません」エラーを解消
✅ 各ユーザーの生活パターンに最適化された予測
✅ 自動訓練により手動操作不要

**制限:**
⚠️ モデルはメモリ上に保存（コンテナ再起動で消失）
⚠️ ユーザー数増加によるメモリ使用量の増加

将来的にCloud Storageへのモデル永続化を検討します。
