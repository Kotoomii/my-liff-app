# 50件データでも予測値が偏る問題の根本原因

## 問題の症状

- データ数: **50件**（十分な量）
- 予測値: **10.8, 10.7, 10.7, 10.7...** と固定
- タブレット上で同じ値が表示され続ける

## 根本原因

### **訓練時と予測時で特徴量の作り方が異なる**

これが最大の問題です。

#### 訓練時（Walk Forward Validation）
```python
# create_features_for_activity() を使用
# 過去24時間の実データから統計特徴量を計算
features = {
    'hist_avg_frustration': actual_data['NASA_F'].mean(),  # 実データ
    'hist_std_frustration': actual_data['NASA_F'].std(),   # 実データ
    'hist_lorenz_mean': actual_data['lorenz_mean'].mean(), # 実データ
    # ... すべて実データから計算
}
```

#### 予測時（タブレット表示用）
```python
# create_features_for_new_activity() を使用
# 固定値を使用
features = {
    'hist_avg_frustration': 10.0,  # 固定値！
    'hist_std_frustration': 2.0,   # 固定値！
    'hist_lorenz_mean': 8000.0,    # 固定値！
    # ... すべて固定値
}
```

### **結果**

- 訓練時: 多様な特徴量 → モデルは多様なパターンを学習
- 予測時: 常に同じ特徴量 → **同じ予測値を返す**

モデルは正しく訓練されているが、予測時に「いつも同じ入力」を与えているため、「いつも同じ出力」が返ってくる。

## 修正内容

### 新しいメソッド: `predict_with_history()`

訓練時と同じ方法で特徴量を作成する新しい予測メソッドを実装：

```python
def predict_with_history(self, activity_category, duration,
                        current_time, historical_data):
    """
    過去の履歴データを使用して予測

    訓練時と同じ特徴量作成方法を使用するため、
    予測値に多様性が生まれる
    """
    # 過去24時間のデータから統計特徴量を計算
    recent_data = historical_data[
        historical_data['Timestamp'] >= current_time - timedelta(hours=24)
    ]

    features = {
        'hist_avg_frustration': recent_data['NASA_F'].mean(),  # 実データ
        'hist_std_frustration': recent_data['NASA_F'].std(),   # 実データ
        # ... すべて実データから計算
    }

    return prediction
```

### タイムラインAPIの修正

```python
# 修正前
prediction_result = predictor.predict_single_activity(
    activity_category=row.get('CatSub'),
    duration=row.get('Duration'),
    current_time=row.get('Timestamp')
)

# 修正後
prediction_result = predictor.predict_with_history(
    activity_category=row.get('CatSub'),
    duration=row.get('Duration'),
    current_time=row.get('Timestamp'),
    historical_data=df_enhanced  # 履歴データを渡す
)
```

## 期待される改善

### Before（修正前）
```
予測値: 10.7, 10.7, 10.7, 10.8, 10.7...
理由: 常に同じ特徴量（固定値）を入力
```

### After（修正後）
```
予測値: 8.2, 12.5, 9.8, 15.3, 7.1...
理由: 実際の履歴データから特徴量を計算
```

## 他の要因（修正済み）

### 1. 活動カテゴリのエンコーディング

**問題:**
```python
# エンコーダーのキーが間違っていた
if 'CatSub' in self.encoders:  # 誤り
    encoded = self.encoders['CatSub'].transform([activity])
```

**修正:**
```python
# 正しいキー名を使用
if 'current_activity' in self.encoders:  # 正しい
    encoded = self.encoders['current_activity'].transform([activity])
```

### 2. 未知の活動カテゴリの処理

**修正:**
```python
try:
    encoded = self.encoders['current_activity'].transform([activity])[0]
except ValueError:
    # 未知のカテゴリは'その他'として扱う
    encoded = self.encoders['current_activity'].transform(['その他'])[0]
```

## データ統計APIの追加

データの状態を確認できるAPIエンドポイントも追加：

```bash
curl "https://your-app.run.app/api/data/stats?user_id=default"
```

レスポンス例：
```json
{
  "nasa_f_stats": {
    "count": 50,
    "mean": 10.2,
    "std": 3.5,
    "min": 5.0,
    "max": 18.0,
    "unique_values": 12
  },
  "data_quality": {
    "is_sufficient": true,
    "quality_level": "moderate",
    "warnings": []
  }
}
```

## なぜこの問題が起きたか

1. **設計の不整合**
   - 訓練用と予測用で別々の特徴量作成関数を用意
   - 両者の実装が異なることに気づかなかった

2. **デフォルト値の使用**
   - リアルタイム予測では履歴データがないと仮定
   - 固定値で代用していた

3. **テストの不足**
   - 訓練と予測の特徴量が同じかを検証していなかった

## 今後の改善案

### 短期
- ✅ `predict_with_history()`を使用（実装済み）
- ✅ タイムラインAPIを修正（実装済み）

### 中期
- 特徴量作成ロジックを統一
- 訓練時と予測時で同じコードパスを使用

### 長期
- 特徴量エンジニアリングの改善
- より高度な特徴量（時系列パターン等）の追加
- モデルのアンサンブル

## まとめ

**問題:** 訓練時と予測時で特徴量の作り方が異なる
**原因:** 予測時に固定値を使用
**解決:** 履歴データから実際の統計を計算する`predict_with_history()`を実装

これにより、50件のデータがある場合でも、過去の履歴に基づいた多様な予測値が得られるようになります。
