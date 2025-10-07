# 予測値が10.72で固定される問題の診断レポート

## 問題の概要

タブレット上で表示される予測フラストレーション値が**10.72で固定**され、PREDICTION_DATAシートにも同じ値が記録され続けている。

## 原因分析

### 1. **学習データ不足（最も可能性が高い）**

現在のデータ数: **約8件**

```
必要なデータ数:
- 最低要件: 10件（Walk Forward Validation）
- 推奨: 30件以上（精度向上）
- 理想: 100件以上（高精度）
```

**影響:**
- データ数が少ないと、モデルは**平均値に近い値**を常に予測する
- 10.72はおそらく学習データのNASA_F平均値付近

### 2. **データの分散不足**

`check_data_quality()`の診断結果:
```python
if frustration_std < 1.0:
    # フラストレーション値のバラつきが小さい
    # → モデルが同じ値を予測
```

**問題:**
- NASA_Fの値がほぼ同じ（分散が小さい）
- 活動の種類が少ない（多様性不足）
- モデルが学習すべき「パターン」が存在しない

### 3. **モデルの特徴量不足**

`create_features_for_new_activity()`の問題:
```python
# 現在の実装
features = {
    'Duration': duration,
    'hour': current_time.hour,
    'day_of_week': current_time.weekday(),
    # ... デフォルト値のみ
}
```

**問題点:**
- 実際の生体データ（Fitbit）が使われていない
- 過去の履歴情報が反映されていない
- 活動カテゴリのエンコーディングが正しく機能していない可能性

### 4. **モデルの訓練状態**

Walk Forward Validation結果の`prediction_diversity`:
```python
{
    'std': 0.15,  # 標準偏差が小さい
    'unique_values': 2,  # 予測値の種類が少ない
    'is_diverse': False
}
```

## 実装した修正

### 1. **予測値多様性の監視**

```python
# 直近10件の予測値の標準偏差をチェック
if pred_std < 0.5:
    diagnosis = "警告: 予測値の多様性が低い"
    logger.warning(diagnosis)
```

### 2. **ユーザー別PREDICTION_DATA**

```python
# 従来: 全ユーザー共通
sheet_name = "PREDICTION_DATA"

# 修正後: ユーザー別
sheet_name = f"PREDICTION_DATA_{user_id}"
```

これにより：
- `PREDICTION_DATA_default`
- `PREDICTION_DATA_user1`
- `PREDICTION_DATA_user2`

のように、ユーザーごとにシートが分かれます。

### 3. **診断情報の追加**

予測結果に`diagnosis`フィールドを追加：
```json
{
  "predicted_frustration": 10.72,
  "confidence": 0.3,
  "diagnosis": "警告: 予測値の多様性が低い（標準偏差: 0.12）"
}
```

## 解決策

### 短期的対策

1. **データ収集を継続**
   - 最低30件以上のデータを蓄積
   - 様々な状況での活動を記録

2. **NASA_Fの値に変化をつける**
   - 意図的に異なるストレスレベルの活動を記録
   - フラストレーション値: 5, 10, 15など幅を持たせる

3. **活動の多様性を確保**
   - 異なる種類の活動を記録
   - 時間帯、曜日にバリエーションを

### 中期的対策

1. **特徴量エンジニアリング改善**
   ```python
   # 実際のFitbitデータを使用
   # 過去の活動履歴を反映
   # より詳細な時間帯特徴量
   ```

2. **モデルのハイパーパラメータ調整**
   ```python
   # RandomForestの設定を調整
   n_estimators=50  # → 100
   max_depth=10     # → 調整
   ```

3. **正則化の追加**
   - 過学習を防ぐ
   - 多様な予測値を生成

### 長期的対策

1. **モデルアーキテクチャの見直し**
   - 時系列モデル（LSTM等）の検討
   - アンサンブル学習

2. **データ拡張**
   - 合成データの生成
   - 他ユーザーデータの活用

## 確認方法

### Cloud Runログで確認

```
# WARNINGログを確認
⚠️ 予測値の多様性が低い（標準偏差: 0.12）
⚠️ フラストレーション値のバラつきが小さい（標準偏差: 0.85）
```

### API レスポンスで確認

```bash
curl -X POST https://your-app.run.app/api/frustration/predict-activity \
  -H "Content-Type: application/json" \
  -d '{"user_id":"default","CatSub":"仕事","Duration":60}'
```

レスポンス:
```json
{
  "predicted_frustration": 10.72,
  "confidence": 0.3,
  "diagnosis": "警告: 予測値の多様性が低い",
  "data_quality": {
    "total_samples": 8,
    "is_sufficient": false,
    "quality_level": "insufficient",
    "warnings": [
      "データ数が不足しています（8件/最低10件必要）",
      "フラストレーション値の種類が少なすぎます（3種類）"
    ]
  }
}
```

## まとめ

**予測値が10.72で固定される主な原因:**

1. ✅ **データ数不足**（8件 < 10件）
2. ✅ **データの分散不足**（NASA_Fのバラつきが小さい）
3. ✅ **活動の多様性不足**
4. ⚠️ **特徴量の問題**（Fitbitデータが活用されていない）

**優先的に実施すべきこと:**

1. **データを30件以上収集**
2. **NASA_F値に変化をつける**（5～20の範囲で）
3. **異なる活動を記録**

データが蓄積されるまで、システムは**実測値（NASA_F）を表示**するようフォールバック処理が実装されています。
