# モデル切り替えガイド

このガイドでは、RandomForestとLinearRegressionを簡単に切り替える方法を説明します。

## モデルの切り替え方法

### 1. config.pyを編集

`config.py`の85-86行目を編集します:

```python
# RandomForestを使用する場合（デフォルト）
MODEL_TYPE = 'RandomForest'

# LinearRegressionを使用する場合
MODEL_TYPE = 'Linear'
```

### 2. アプリケーションを再起動

変更を反映するには、Flaskアプリケーションを再起動してください:

1. **現在実行中のアプリを停止**: ターミナルで `Ctrl + C`
2. **もう一度起動**: `python main.py` または `python3 main.py`

### 3. モデルの確認

アプリケーション起動時のログで、どのモデルが使用されているか確認できます:

```
INFO: RandomForestRegressorモデルを使用します
```

または

```
INFO: LinearRegressionモデルを使用します
```

## モデルの特徴

### RandomForest（デフォルト）
- **メリット**:
  - 非線形な関係を学習できる
  - 複雑なパターンを捉えられる
  - 過学習しにくい
- **デメリット**:
  - 訓練時間が長い
  - DiCEで反実仮想例が見つかりにくい場合がある

### LinearRegression
- **メリット**:
  - 訓練が高速
  - DiCEで反実仮想例が見つかりやすい
  - 解釈しやすい（係数で直接影響度がわかる）
- **デメリット**:
  - 線形な関係しか学習できない
  - 複雑なパターンは捉えられない

## DiCEエラーのトラブルシューティング

`No counterfactuals found` エラーが出る場合:

1. **Linearモデルを試す**: `config.py`で `MODEL_TYPE = 'Linear'` に変更
2. **アプリを再起動**
3. **ログを確認**: DiCEが反実仮想例を生成できるか確認

Linearモデルで動作する場合、モデルの複雑さが原因の可能性があります。

## 元に戻す方法

RandomForestに戻すには:

```python
# config.py
MODEL_TYPE = 'RandomForest'
```

再起動すれば元に戻ります。

## 注意事項

- モデルを切り替えると、既存の保存済みモデルファイルは使用できなくなります
- 初回起動時に自動的に新しいモデルで再訓練されます
- データが同じであれば、予測精度は大きく変わりません（データの特性による）
