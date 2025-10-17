# シート再作成ガイド

PREDICTION_DATAシートやDAILY_SUMMARYシートが正しく表示されない場合（斜めにデータが入っている、ヘッダーが欠けているなど）、シートを再作成できます。

## 問題が発生した場合

### 症状:
- PREDICTION_DATA_defaultシートのヘッダーが正しく表示されない
- データが斜めに入力されている（2行目がB列から、3行目がC列から...など）
- 列がずれている

### 原因:
- Google Sheets APIの`append_row()`が予期しない動作をした
- シート構造が古い仕様のまま

## 解決方法

### 方法1: APIエンドポイントを使用（推奨）

アプリを起動後、以下のAPIを呼び出してシートを再作成できます。

#### PREDICTION_DATAシートの再作成:

```bash
curl -X POST http://localhost:8080/api/sheets/recreate-prediction \
  -H "Content-Type: application/json" \
  -d '{"user_id": "default"}'
```

#### DAILY_SUMMARYシートの再作成:

```bash
curl -X POST http://localhost:8080/api/sheets/recreate-daily-summary \
  -H "Content-Type: application/json" \
  -d '{"user_id": "default"}'
```

### 方法2: Pythonコンソールから実行

```python
from sheets_connector import SheetsConnector

connector = SheetsConnector()

# PREDICTION_DATAシートを再作成
connector.recreate_prediction_sheet(user_id='default')

# DAILY_SUMMARYシートを再作成
connector.recreate_daily_summary_sheet(user_id='default')
```

### 方法3: 手動でGoogle Sheetsから削除

1. Google Sheetsを開く
2. 問題のあるシート（PREDICTION_DATA_default, DAILY_SUMMARY_default）を右クリック
3. 「削除」を選択
4. アプリを再起動すると、自動的に新しい構造でシートが作成されます

## 新しいシート構造

### PREDICTION_DATA_default
| 列 | 説明 |
|----|------|
| Timestamp | タイムスタンプ |
| Activity | 活動名 |
| Duration | 活動時間（分） |
| **ActualFrustration** | **実測F値（1-20）** |
| **PredictedFrustration** | **予測F値（1-20）** |
| PredictionError | 予測誤差（絶対値） |
| Confidence | 信頼度（0-1） |
| Notes | 備考 |

### DAILY_SUMMARY_default
| 列 | 説明 |
|----|------|
| Date | 日付 |
| **AvgFrustration** | **平均F値** |
| **MinFrustration** | **最小F値** |
| **MaxFrustration** | **最大F値** |
| **ActivityCount** | **活動数** |
| TotalDuration | 合計時間（分） |
| UniqueActivities | ユニークな活動数 |
| Notes | 備考 |

## 注意事項

1. **データのバックアップ**: シート再作成は既存データを**削除**します。重要なデータは事前にバックアップしてください。

2. **レート制限**: Google Sheets APIには1分あたり60リクエストの制限があります。エラーが出た場合は5分待ってから再試行してください。

3. **アプリの再起動**: シート再作成後、アプリケーションを再起動することを推奨します。

## トラブルシューティング

### エラー: "シート再作成に失敗しました"
- Google Sheets APIの認証情報を確認してください
- スプレッドシートの共有設定を確認してください（サービスアカウントに編集権限があるか）

### エラー: "Rate limit exceeded"
- 5分待ってから再試行してください
- アプリを複数起動していないか確認してください

### データが再び斜めに入る
- `append_row()`ではなく`update()`を使用するように修正されています
- 最新のコードにアップデートしてください

## 実測値の記録について

PREDICTION_DATAシートには**ActualFrustration（実測F値）**列が追加されています。

### 実測値の記録方法:

1. ユーザーが活動を記録する際、NASA-TLXのF値（フラストレーション値）を入力
2. その値が`actual_frustration`として`save_prediction_data()`に渡される
3. 予測値と実測値が同じ行に記録される
4. PredictionError（予測誤差）が自動計算される

### データの見方:
- **ActualFrustration**: ユーザーが実際に感じたフラストレーション値
- **PredictedFrustration**: モデルが予測したフラストレーション値
- **PredictionError**: `|ActualFrustration - PredictedFrustration|`

これにより、モデルの精度を日々追跡できます！
