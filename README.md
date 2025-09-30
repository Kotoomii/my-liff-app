# 🧠 フラストレーション値予測・反実仮想説明システム

**NASA-TLXフラストレーション値**に特化した予測システムです。**Walk Forward Validation**と**DiCE（反実仮想説明）**を活用し、行動変更タイミングでの予測と改善提案を提供します。

## 🎯 システム概要

### 💡 核心機能
- **フラストレーション値予測**: NASA_F（フラストレーション）に特化した機械学習予測
- **Walk Forward Validation**: 過去24時間のデータで学習し、現在値を予測する時系列手法
- **行動変更タイミング予測**: 活動が変わるタイミングで自動予測を実行
- **DiCE反実仮想説明**: 過去24時間の行動から改善提案を自動生成
- **LLM自然言語フィードバック**: DiCE結果を温かく具体的なアドバイスに変換
- **定期フィードバック**: 毎朝7:30と夜21:00に自動でフィードバックを配信

### 🔧 技術特徴
- **Fitbit統計量化**: 15分ごとのローレンツプロット面積を行動長さごとに統計処理
- **行動単位分析**: 最小15分の活動区切りで詳細分析
- **時系列最適化**: Walk Forward Validationによる現実的な予測性能評価
- **Web API**: RESTful APIによる全機能へのアクセス
- **GCP対応**: Google App Engineでの本格運用対応

## 📊 システム構成

```
┌─ 活動データ（Google Sheets）
│  └─ NASA_F, CatSub, Duration, Timestamp
│
├─ Fitbitデータ（Google Sheets）  
│  └─ Lorenz_Area (15分ごと)
│
├─ フラストレーション予測モデル
│  ├─ データ前処理・統計量化
│  ├─ Walk Forward Validation学習
│  └─ 行動変更タイミング予測
│
├─ DiCE反実仮想説明
│  ├─ 過去24時間行動分析
│  ├─ 代替行動提案生成
│  └─ 改善効果計算
│
├─ LLM自然言語フィードバック
│  ├─ DiCE結果の自然言語変換
│  ├─ 朝のブリーフィング生成
│  └─ 夜のサマリー生成
│
└─ Webダッシュボード・API
   ├─ リアルタイム予測API
   ├─ DiCE分析API
   ├─ タイムライン可視化
   └─ 定期フィードバック配信
```

## 🚀 セットアップ

### 1. ローカル開発環境

#### 環境設定
```bash
# 仮想環境の作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存関係のインストール
pip install -r requirements.txt
```

#### Google APIの設定
1. [Google Cloud Console](https://console.cloud.google.com/) でプロジェクトを作成
2. 以下のAPIを有効化:
   - Google Sheets API
   - Google Drive API
3. サービスアカウントを作成し、キーファイル `service-account-key.json` をダウンロード
4. `.env` ファイルを作成:

```bash
# .env
SPREADSHEET_ID=your_spreadsheet_id_here
GOOGLE_APPLICATION_CREDENTIALS=service-account-key.json
OPENAI_API_KEY=your_openai_api_key  # LLMフィードバック用（任意）
FLASK_ENV=development
```

#### Googleスプレッドシートの準備（複数ユーザー対応）
スプレッドシートに以下のシートを作成:

**活動データシート（複数ユーザー対応）**:
- **LINEユーザーIDシート**（例：`U1234567890abcdef`）
- **デフォルトシート**（名前：`default`）
- 各シートの列構成:
  - `Timestamp`: タイムスタンプ（YYYY-MM-DD HH:MM:SS）
  - `CatSub`: 活動カテゴリ
  - `Duration`: 継続時間（分）
  - `NASA_F`: **フラストレーション値**（1-20スケール）⚠️
  - その他NASA-TLX項目（NASA_M, NASA_P, NASA_T, NASA_O, NASA_E）

**生体データシート（複数ユーザー対応）**:
- **Fitbitデータシート**（例：`kotoomi_Fitbit-data-01`, `kotoomi_Fitbit-data-02`）
- 各シートの列構成:
  - `Timestamp`: タイムスタンプ（15分間隔）
  - `Lorenz_Area`: ローレンツプロット面積

**固定予定シート**（名前: "FIXED_PLANS"）:
- `Date`: 日付（YYYY-MM-DD）
- `Activity`: 活動名
- `StartTime`: 開始時刻（HH:MM）
- `EndTime`: 終了時刻（HH:MM）
- `UserID`: ユーザーID
- `Fixed`: 固定フラグ（"Yes"）

**ユーザー・シート対応設定**:
```python
# main.py で設定
{
    'user_id': 'user1', 
    'name': 'ユーザー1', 
    'icon': '👨',
    'activity_sheet': 'U1234567890abcdef',  # LINEユーザーID
    'fitbit_sheet': 'kotoomi_Fitbit-data-01'
}
```

#### ローカル実行
```bash
python main.py
```

ブラウザで `http://localhost:8080` にアクセス

#### 利用可能なUIインターフェース
- **メインダッシュボード**: `http://localhost:8080/` - 過去24時間DiCE結果の可視化
- **スマートミラー**: `http://localhost:8080/mirror` - 完全自動運転・タッチレス操作  
- **タブレットUI**: `http://localhost:8080/tablet` - 手動ユーザー選択・日次平均表示
- **推移分析**: `http://localhost:8080/trends` - ユーザー別フラストレーション値推移確認

#### タブレットUI の新機能
- **複数ユーザー選択**: ドロップダウンから簡単切り替え
- **3時間間隔タイムライン**: 0,3,6,9,12,15,18,21時表示
- **1-20スケール対応**: 1-6(低)、7-13(中)、14-20(高)の色分け
- **DiCE提案タイムライン**: 実際の活動の下に改善提案を表示
- **時間軸ベース表示**: 活動の開始時刻と継続時間を正確に可視化

### 2. Google Cloud Platform 詳細移行手順（Webコンソールベース）

#### ステップ1: 既存プロジェクト確認・設定

1. **既存プロジェクト（wakayama-erserch）の確認**
   - [Google Cloud Console](https://console.cloud.google.com/) にアクセス
   - 左上のプロジェクトセレクターをクリック
   - 「wakayama-erserch」を選択
   - プロジェクトIDをメモ（通常は `wakayama-erserch` または類似）

2. **Google Cloud SDK インストール（任意、デプロイ時に必要）**
   ```bash
   # macOS/Linux
   curl https://sdk.cloud.google.com | bash
   exec -l $SHELL
   
   # Windows PowerShell
   (New-Object Net.WebClient).DownloadFile("https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe", "$env:Temp\GoogleCloudSDKInstaller.exe")
   & $env:Temp\GoogleCloudSDKInstaller.exe
   
   # または https://cloud.google.com/sdk/docs/install からダウンロード
   ```

3. **認証設定（SDK使用時）**
   ```bash
   # Google アカウントでログイン
   gcloud auth login
   
   # プロジェクト設定
   gcloud config set project wakayama-erserch
   
   # 現在の設定確認
   gcloud config list
   ```

### Cloud Run デプロイ手順（wakayama-erserch プロジェクト）

既存の「wakayama-erserch」プロジェクトを使用してWebコンソール経由でCloud Runにデプロイを行います。

#### Cloud Run vs App Engine
**Cloud Run の利点:**
- コンテナベースでより柔軟
- オートスケーリングが優秀（0インスタンスまで縮小可能）
- 使用分のみ課金でコストが安い
- より現代的なアーキテクチャ

#### ステップ1: 必要ファイル作成

**Dockerfile作成:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# システムの依存関係をインストール
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Pythonの依存関係をインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションファイルをコピー
COPY . .

# ポート設定
ENV PORT 8080
EXPOSE $PORT

# アプリケーション起動
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
```

#### ステップ2: 必要なAPI有効化（Webコンソール）

1. **APIライブラリにアクセス**
   - [APIライブラリ](https://console.cloud.google.com/apis/library) にアクセス
   - プロジェクトが「wakayama-erserch」になっていることを確認

2. **Cloud Run Admin API の有効化**
   - 検索ボックスに「Cloud Run Admin API」と入力
   - 「Cloud Run Admin API」をクリック
   - 「有効にする」ボタンをクリック

3. **Container Registry API の有効化**
   - 検索ボックスに「Container Registry API」と入力
   - 「Container Registry API」をクリック
   - 「有効にする」ボタンをクリック

4. **Google Sheets API の有効化**
   - 検索ボックスに「Google Sheets API」と入力
   - 「Google Sheets API」をクリック  
   - 「有効にする」ボタンをクリック

5. **Google Drive API の有効化**
   - 検索ボックスに「Google Drive API」と入力
   - 「Google Drive API」をクリック
   - 「有効にする」ボタンをクリック

6. **Cloud Build API の有効化**
   - 検索ボックスに「Cloud Build API」と入力
   - 「Cloud Build API」をクリック
   - 「有効にする」ボタンをクリック

7. **有効化確認**
   - [API とサービス > 有効な API](https://console.cloud.google.com/apis/dashboard) で確認

#### ステップ3: 既存サービスアカウントの設定（Webコンソール）

既存の `wakayama-research@appspot.gserviceaccount.com` を使用します。

1. **IAM と管理にアクセス**
   - [IAM と管理 > サービス アカウント](https://console.cloud.google.com/iam-admin/serviceaccounts) にアクセス
   - プロジェクトが「wakayama-erserch」になっていることを確認

2. **既存サービスアカウントの権限確認・追加**
   - `wakayama-research@appspot.gserviceaccount.com` をクリック
   - 「権限」タブで現在の権限を確認
   - 必要に応じて以下の権限を追加:
     - `Cloud Run 開発者` (Cloud Run デプロイ・実行用)
     - `ストレージ オブジェクト閲覧者` (Container Registry用)
     - 既存の権限がApp Engine管理者等であれば、そのまま利用可能

3. **キーファイル生成・ダウンロード**
   - 同じサービスアカウントの「キー」タブをクリック
   - 既存のキーがあれば使用、なければ新規作成:
     - 「キーを追加」> 「新しいキーを作成」をクリック
     - **キーのタイプ**: `JSON` を選択
     - 「作成」をクリック
   - ダウンロードされたJSONファイルを `service-account-key.json` にリネーム
   - プロジェクトフォルダに配置

5. **本番環境用権限設定（推奨）**
   ```text
   本番環境では以下の最小権限に変更することを推奨:
   - Cloud Run 開発者
   - ストレージ オブジェクト管理者
   - Cloud SQL クライアント（必要に応じて）
   ```

#### ステップ4: Cloud Run デプロイ（Webコンソール）

1. **Cloud Run にアクセス**
   - [Cloud Run](https://console.cloud.google.com/run) にアクセス
   - プロジェクトが「wakayama-erserch」になっていることを確認

2. **新しいサービス作成**
   - 「サービスの作成」をクリック
   - 「ソースリポジトリから継続的にデプロイする」を選択
   - または「1つのコンテナイメージを既存のコンテナイメージからデプロイ」を選択

3. **ソースの設定**
   - **GitHub接続**を選択（推奨）
   - リポジトリを接続し、ブランチを選択
   - **ビルドタイプ**: Dockerfile を選択

4. **サービス設定**
   - **サービス名**: `frustration-system`
   - **リージョン**: `asia-northeast1`（東京）
   - **CPU 割り当て**: リクエスト時のみCPUを割り当て（コスト削減）
   - **認証**: 認証が必要（未認証の呼び出しを許可のチェックを外す）

5. **環境変数の設定**
   - 「コンテナ、変数とシークレット、接続、セキュリティ」を展開
   - 以下の環境変数を追加:
     ```
     SPREADSHEET_ID=実際のスプレッドシートID
     GOOGLE_APPLICATION_CREDENTIALS=/app/service-account-key.json
     OPENAI_API_KEY=実際のOpenAI APIキー（任意）
     ```

6. **リソース設定**
   - **CPU**: 1
   - **メモリ**: 512 MiB
   - **最大同時リクエスト数**: 80
   - **タイムアウト**: 300秒

7. **デプロイ実行**
   - 「作成」をクリック
   - ビルドとデプロイの進行状況を確認

#### ステップ5: Google Sheets の設定

1. **スプレッドシート共有設定**
   - Google Sheets でスプレッドシートを開く
   - 「共有」ボタンをクリック
   - 既存のサービスアカウントのメールアドレスを追加（未追加の場合）
     ```
     wakayama-research@appspot.gserviceaccount.com
     ```
   - 権限を「編集者」に設定

2. **スプレッドシートID取得**
   - URL から ID を抽出
   ```
   https://docs.google.com/spreadsheets/d/SPREADSHEET_ID_HERE/edit
   ```

#### ステップ6: 設定ファイル準備

1. **app.yaml 作成・編集**
   ```yaml
   runtime: python39
   
   env_variables:
     # 実際のスプレッドシートIDに変更
     SPREADSHEET_ID: "1abc123def456ghi789jkl"
     
     # サービスアカウントキーファイル
     GOOGLE_APPLICATION_CREDENTIALS: "service-account-key.json"
     
     # OpenAI API Key（任意、LLMフィードバック用）
     OPENAI_API_KEY: "sk-your-openai-api-key"
     
     # 本番環境設定
     FLASK_ENV: "production"
   
   automatic_scaling:
     min_instances: 1
     max_instances: 10
     target_cpu_utilization: 0.6
   
   resources:
     cpu: 1
     memory_gb: 0.5
     disk_size_gb: 10
   ```

2. **requirements.txt 確認**
   ```bash
   # 依存関係が最新か確認
   pip list --outdated
   
   # requirements.txt 生成
   pip freeze > requirements.txt
   ```

#### ステップ5: Cloud Build 設定（Webコンソール）

1. **Cloud Build にアクセス**
   - [Cloud Build](https://console.cloud.google.com/cloud-build) にアクセス
   - プロジェクトが「wakayama-erserch」になっていることを確認

2. **トリガー作成（任意、継続的デプロイ用）**
   - 「トリガー」タブをクリック
   - 「トリガーを作成」をクリック
   - 名前: `app-deploy-trigger`
   - ソース: GitHub または Cloud Source Repositories
   - ビルド構成: `app.yaml` を選択

#### ステップ6: App Engine デプロイ（Webコンソール）

1. **App Engine にアクセス**
   - [App Engine](https://console.cloud.google.com/appengine) にアクセス
   - プロジェクトが「wakayama-erserch」になっていることを確認

2. **ソースコードのアップロード準備**
   - プロジェクトファイルをZIP形式で圧縮
   - 必要ファイル: `main.py`, `app.yaml`, `requirements.txt`, `service-account-key.json`

3. **デプロイ実行**
   - 「デプロイ」ボタンをクリック
   - 「ソースをアップロード」を選択
   - ZIPファイルをアップロード
   - 「デプロイ」をクリック

4. **デプロイ状況確認**
   - [App Engine > バージョン](https://console.cloud.google.com/appengine/versions) で確認
   - デプロイ完了まで数分待機

#### ステップ7: ドメイン・URL設定（Webコンソール）

1. **アプリケーションURL確認**
   - [App Engine > 設定](https://console.cloud.google.com/appengine/settings) にアクセス
   - デフォルトURL: `https://wakayama-erserch.appspot.com`

2. **カスタムドメイン設定（任意）**
   - [App Engine > 設定 > カスタム ドメイン](https://console.cloud.google.com/appengine/settings/domains) にアクセス
   - 「カスタム ドメインを追加」をクリック
   - 所有するドメインを入力・認証

#### ステップ8: 監視・ログ設定（Webコンソール）

1. **ログ確認**
   - [Cloud Logging](https://console.cloud.google.com/logs) にアクセス
   - リソース: `GAE Application` を選択
   - アプリケーションログを確認

2. **監視設定**
   - [Monitoring](https://console.cloud.google.com/monitoring) にアクセス
   - 「ダッシュボードを作成」をクリック
   - App Engineメトリクスを追加

#### ステップ9: セキュリティ・アクセス制御（Webコンソール）

1. **IAP（Identity-Aware Proxy）設定**
   - [Security > Identity-Aware Proxy](https://console.cloud.google.com/security/iap) にアクセス
   - App Engine アプリを選択
   - IAP を有効化（アクセス制御が必要な場合）

2. **ファイアウォール設定**
   - [App Engine > ファイアウォール](https://console.cloud.google.com/appengine/firewall) にアクセス
   - 必要に応じてアクセス制限を設定

#### ステップ10: 運用・メンテナンス（Webコンソール）

1. **バージョン管理**
   - [App Engine > バージョン](https://console.cloud.google.com/appengine/versions) で管理
   - 新バージョンデプロイ時の切り替え設定

2. **自動スケーリング調整**
   - [App Engine > インスタンス](https://console.cloud.google.com/appengine/instances) で確認
   - `app.yaml` の scaling 設定を調整

3. **コスト監視**
   - [Billing](https://console.cloud.google.com/billing) でコスト確認
   - 予算アラートの設定

#### Cloud Run アプリケーション動作確認

1. **Webコンソールでの確認**
   - [Cloud Run](https://console.cloud.google.com/run) のダッシュボードで稼働状況確認
   - デプロイしたサービス「frustration-system」をクリック
   - 「URL」をクリックしてアプリケーションにアクセス

2. **各機能への直接アクセス**
   ```
   https://frustration-system-xxx-an.a.run.app/          # ダッシュボード
   https://frustration-system-xxx-an.a.run.app/mirror    # スマートミラー
   https://frustration-system-xxx-an.a.run.app/tablet    # タブレット
   https://frustration-system-xxx-an.a.run.app/trends    # 推移分析
   ```
   ※ `xxx` の部分は実際のCloud Runサービス固有の文字列

2. **ログ確認**
   ```bash
   # リアルタイムログ
   gcloud app logs tail -s default
   
   # エラーログのみ
   gcloud app logs tail -s default --level=error
   
   # 過去のログ
   gcloud app logs read --limit=50
   ```

#### ステップ9: トラフィック管理

```bash
# すべてのトラフィックを新バージョンに
gcloud app services set-traffic default --splits=v1=100

# 段階的リリース（50%ずつ）
gcloud app services set-traffic default --splits=v1=50,v2=50

# バージョン削除
gcloud app versions delete v1
```

#### ステップ10: 継続的デプロイ設定（任意）

1. **自動デプロイスクリプト作成（deploy.sh）**
   ```bash
   #!/bin/bash
   
   # エラー発生時は停止
   set -e
   
   echo "🚀 Google Cloud へのデプロイを開始します..."
   
   # プロジェクトID確認
   PROJECT_ID=$(gcloud config get-value project)
   echo "📋 プロジェクト: $PROJECT_ID"
   
   # 環境変数チェック
   if [ -z "$SPREADSHEET_ID" ]; then
       echo "❌ SPREADSHEET_ID 環境変数が設定されていません"
       exit 1
   fi
   
   # サービスアカウントキー存在確認
   if [ ! -f "service-account-key.json" ]; then
       echo "❌ service-account-key.json が見つかりません"
       exit 1
   fi
   
   # app.yaml の環境変数更新
   sed -i.bak "s/SPREADSHEET_ID_PLACEHOLDER/$SPREADSHEET_ID/g" app.yaml
   
   if [ -n "$OPENAI_API_KEY" ]; then
       sed -i.bak "s/OPENAI_API_KEY_PLACEHOLDER/$OPENAI_API_KEY/g" app.yaml
   fi
   
   # デプロイ実行
   echo "📦 デプロイ実行中..."
   gcloud app deploy app.yaml --quiet
   
   # URL表示
   echo "✅ デプロイ完了!"
   echo "🌐 アプリケーションURL:"
   gcloud app browse --no-launch-browser
   
   # バックアップファイル削除
   rm -f app.yaml.bak
   ```

2. **使用方法**
   ```bash
   # 実行権限付与
   chmod +x deploy.sh
   
   # 環境変数設定してデプロイ
   export SPREADSHEET_ID="your_actual_spreadsheet_id"
   export OPENAI_API_KEY="your_openai_api_key"  # 任意
   ./deploy.sh
   ```

#### 💡 コスト最適化のヒント

1. **自動スケーリング設定**
   ```yaml
   automatic_scaling:
     min_instances: 0  # コスト削減
     max_instances: 5
     target_cpu_utilization: 0.8
   ```

2. **リソース制限**
   ```yaml
   resources:
     cpu: 0.5  # 最小CPU
     memory_gb: 0.3  # 最小メモリ
   ```

3. **定期バックアップ**
   ```bash
   # Cloud Scheduler でのバックアップ自動化
   gcloud scheduler jobs create http backup-job \
       --schedule="0 2 * * *" \
       --uri="https://YOUR_PROJECT_ID.appspot.com/api/backup" \
       --http-method=POST
   ```

## 📡 API仕様

### 基本分析API

#### POST /api/frustration/predict
フラストレーション値予測

**Request:**
```json
{
    "user_id": "default",
    "timestamp": "2023-09-11T15:30:00"  // 任意
}
```

**Response:**
```json
{
    "status": "success",
    "user_id": "default",
    "prediction": {
        "predicted_frustration": 65.2,
        "actual_frustration": 70.0,
        "activity": "仕事",
        "timestamp": "2023-09-11T15:30:00",
        "confidence": 0.85
    },
    "activity_change_timestamps": ["2023-09-11T09:00:00", "..."],
    "model_performance": {
        "walk_forward_rmse": 8.5,
        "walk_forward_r2": 0.72
    }
}
```

#### POST /api/frustration/dice-analysis
DiCE反実仮想分析

**Request:**
```json
{
    "user_id": "default",
    "timestamp": "2023-09-11T15:30:00",
    "lookback_hours": 24
}
```

**Response:**
```json
{
    "status": "success",
    "dice_analysis": {
        "type": "activity_counterfactual",
        "total_improvement": 45.8,
        "timeline": [
            {
                "timestamp": "2023-09-11T09:00:00",
                "original_activity": "仕事",
                "suggested_activity": "軽い運動",
                "frustration_reduction": 15.2
            }
        ],
        "top_suggestions": [
            "09:00 - 「仕事」を「軽い運動」に変更すると15.2点改善",
            "14:00 - 「会議」を「休憩」に変更すると12.8点改善"
        ]
    }
}
```

#### POST /api/feedback/generate
LLM自然言語フィードバック生成

**Request:**
```json
{
    "user_id": "default",
    "type": "evening"  // "morning" または "evening"
}
```

**Response:**
```json
{
    "status": "success",
    "feedback": {
        "type": "evening_summary",
        "main_feedback": "今日もお疲れさまでした。全体的に45.8点のフラストレーション改善機会がありました。特に午前中の仕事時間に軽い運動を取り入れることで、大きなストレス軽減が期待できそうです。明日はより快適な一日になりますように。",
        "achievements": ["バランスの取れた一日を過ごせました"],
        "tomorrow_recommendations": ["朝の軽い運動を取り入れてみてください"]
    }
}
```

### スケジューラーAPI

#### GET /api/scheduler/status
定期フィードバックの状態確認

#### POST /api/scheduler/config
スケジューラー設定更新

**Request:**
```json
{
    "morning_time": "07:30",
    "evening_time": "21:00", 
    "enabled": true
}
```

#### POST /api/scheduler/trigger
手動フィードバック実行

**Request:**
```json
{
    "user_id": "default",
    "type": "evening"
}
```

### その他のAPI

#### GET /api/health
システムヘルスチェック

#### GET /api/users
利用可能ユーザー一覧

#### GET /api/feedback/history?user_id=default&days=7
フィードバック履歴取得

## 🎛️ 使用方法

### 1. 基本的なワークフロー

1. **データ準備**: GoogleスプレッドシートにNASA_F列を含む活動データを入力
2. **自動予測**: システムが行動変更タイミングを検出し、フラストレーション値を予測
3. **DiCE分析**: 過去24時間の行動から改善提案を自動生成
4. **自然言語フィードバック**: LLMが人間にわかりやすいアドバイスを生成
5. **定期配信**: 毎朝・夜に自動でフィードバックを受信

### 2. Walk Forward Validationの特徴

- **現実的な予測**: 未来の情報を使わない、実際の利用シーンに近い学習
- **24時間ウィンドウ**: 過去24時間のデータで現在を予測
- **逐次学習**: 新しいデータが入るたびにモデルを更新

### 3. DiCE反実仮想説明の活用

- **行動単位の提案**: 「10:00の仕事を軽い運動に変更すると15点改善」
- **時間帯別分析**: 朝・午後・夕方・夜の改善ポテンシャル
- **実現可能性考慮**: 時間帯と活動の適合性を考慮した代替案

### 4. キーボードショートカット（Web UI）

- `R`: データ更新
- `S`: 設定モーダル表示

## 🔧 システム管理

### ログの確認（GCP）
```bash
# リアルタイムログ
gcloud app logs tail -s default

# エラーログのみ
gcloud app logs tail -s default --level=error
```

### バージョン管理（GCP）
```bash
# バージョン一覧
gcloud app versions list

# 特定バージョンにトラフィック配分
gcloud app services set-traffic default --splits=v1=100
```

### データバックアップ
```bash
# Google Sheetsデータのエクスポート（手動）
# フィードバック履歴は feedback_history/ フォルダに自動保存
```

## 📋 トラブルシューティング

### よくある問題

#### 認証エラー
- サービスアカウントキーのパス確認
- スプレッドシートの共有設定確認（サービスアカウントに編集権限付与）

#### データ読み込みエラー
- スプレッドシートのシート名がパターンと一致するか確認
- NASA_F列が存在し、数値データが入力されているか確認
- Timestamp列の形式確認（YYYY-MM-DD HH:MM:SS）

#### 予測精度が低い場合
- 学習データが最低10件以上あることを確認
- 活動の多様性を確保（同じ活動ばかりでは学習困難）
- Fitbitデータの欠損値確認

#### LLMフィードバックが生成されない
- OpenAI API keyの設定確認
- API利用制限・クォータ確認
- ルールベースフィードバックにフォールバック（正常動作）

### デバッグモード
```bash
# 詳細ログ出力
export FLASK_ENV=development
python main.py
```

## 🤖 モデル作成・訓練方法

### 基本的なモデル作成フロー

#### 1. データ前処理とフィーチャーエンジニアリング

```python
def preprocess_activity_data(df):
    """活動データの前処理"""
    # 時間特徴量の抽出
    df['hour'] = pd.to_datetime(df['Timestamp']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['Timestamp']).dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    
    # カテゴリデータのエンコーディング
    df = pd.get_dummies(df, columns=['CatMid', 'CatSub'])
    
    # NASA-TLX スコアの正規化
    nasa_cols = ['NASA_M', 'NASA_P', 'NASA_T', 'NASA_O', 'NASA_E']
    df[nasa_cols] = df[nasa_cols] / 20.0  # 0-1スケールに正規化
    
    # 活動変化の検出
    df['activity_change'] = (df['CatSub'] != df['CatSub'].shift(1)).astype(int)
    
    return df
```

#### 2. Fitbitデータとの統合・特徴量生成

```python
def aggregate_fitbit_by_activity(activity_df, fitbit_df):
    """活動時間帯ごとのFitbitデータ集約"""
    enhanced_df = activity_df.copy()
    
    for idx, row in activity_df.iterrows():
        start_time = pd.to_datetime(row['Timestamp'])
        end_time = start_time + pd.Timedelta(minutes=row['Duration'])
        
        # 活動時間帯のFitbitデータを抽出
        mask = (fitbit_df['Timestamp'] >= start_time) & (fitbit_df['Timestamp'] <= end_time)
        fitbit_subset = fitbit_df[mask]
        
        if not fitbit_subset.empty:
            # ローレンツプロット統計量の計算
            enhanced_df.loc[idx, 'lorenz_mean'] = fitbit_subset['Lorenz_Area'].mean()
            enhanced_df.loc[idx, 'lorenz_std'] = fitbit_subset['Lorenz_Area'].std()
            enhanced_df.loc[idx, 'lorenz_max'] = fitbit_subset['Lorenz_Area'].max()
            enhanced_df.loc[idx, 'lorenz_min'] = fitbit_subset['Lorenz_Area'].min()
        else:
            # データがない場合はデフォルト値
            enhanced_df.loc[idx, 'lorenz_mean'] = 0
            enhanced_df.loc[idx, 'lorenz_std'] = 0
            enhanced_df.loc[idx, 'lorenz_max'] = 0
            enhanced_df.loc[idx, 'lorenz_min'] = 0
    
    return enhanced_df
```

#### 3. Walk Forward Validation による時系列予測

```python
def walk_forward_validation_predict(self, X, y, test_size_days=7, min_train_days=30):
    """ウォークフォワード検証による時系列予測"""
    predictions = []
    
    # 時系列順でソート
    X_sorted = X.sort_index()
    y_sorted = y.sort_index()
    
    # 最小訓練期間を確保
    min_train_size = min_train_days * 24  # 1日24時間分のデータ
    test_size = test_size_days * 24
    
    start_idx = min_train_size
    for end_idx in range(start_idx + test_size, len(X_sorted) + 1, test_size):
        # 訓練データ
        train_X = X_sorted.iloc[:end_idx - test_size]
        train_y = y_sorted.iloc[:end_idx - test_size]
        
        # テストデータ
        test_X = X_sorted.iloc[end_idx - test_size:end_idx]
        
        # モデル訓練
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        model.fit(train_X, train_y)
        
        # 予測
        pred = model.predict(test_X)
        
        # 結果を保存
        for i, prediction in enumerate(pred):
            actual_idx = end_idx - test_size + i
            predictions.append({
                'index': X_sorted.index[actual_idx],
                'predicted_frustration': float(prediction),
                'model_version': f'wfv_{end_idx}'
            })
    
    return predictions
```

#### 4. モデル性能評価

```python
def evaluate_model_performance(predictions, actual_values):
    """モデル性能の評価"""
    pred_values = [p['predicted_frustration'] for p in predictions]
    
    mae = mean_absolute_error(actual_values, pred_values)
    mse = mean_squared_error(actual_values, pred_values)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_values, pred_values)
    
    print(f"Walk Forward Validation 結果:")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²: {r2:.3f}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'n_predictions': len(predictions)
    }
```

### リアルタイム予測のためのモデル更新

#### 1. 増分学習による継続的改善

```python
def update_model_with_new_data(self, new_activity_data, new_fitbit_data):
    """新しいデータでモデルを増分更新"""
    
    # 新データの前処理
    processed_activity = self.preprocess_activity_data(new_activity_data)
    enhanced_data = self.aggregate_fitbit_by_activity(processed_activity, new_fitbit_data)
    
    # 特徴量とターゲットの準備
    feature_columns = self.get_feature_columns()
    X_new = enhanced_data[feature_columns]
    y_new = enhanced_data['NASA_F']
    
    # 既存データと結合
    self.training_data = pd.concat([self.training_data, enhanced_data])
    
    # 最新の24時間分のデータでモデル再訓練
    recent_data = self.training_data.tail(24)  # 24時間分
    X_recent = recent_data[feature_columns]
    y_recent = recent_data['NASA_F']
    
    # モデル再訓練
    self.current_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    self.current_model.fit(X_recent, y_recent)
    
    logger.info(f"モデル更新完了: 訓練データ{len(X_recent)}件")
```

#### 2. 予測信頼度の計算

```python
def calculate_prediction_confidence(self, X):
    """予測の信頼度を計算"""
    
    # Random Forestの各決定木から予測を取得
    tree_predictions = []
    for tree in self.current_model.estimators_:
        pred = tree.predict(X)
        tree_predictions.append(pred)
    
    # 予測値の分散を信頼度として利用
    predictions_array = np.array(tree_predictions)
    prediction_std = np.std(predictions_array, axis=0)
    
    # 分散が小さいほど信頼度が高い
    confidence = 1.0 / (1.0 + prediction_std)
    
    return float(confidence[0]) if len(confidence) > 0 else 0.5
```

### 実際の訓練実行手順

#### 1. 初期モデル作成

```bash
# 1. 必要なデータを確認
python -c "
from ml_model import FrustrationPredictor
from sheets_connector import SheetsConnector

# データ取得
sheets = SheetsConnector()
activity_data = sheets.get_activity_data('default')
fitbit_data = sheets.get_fitbit_data('default')

print(f'活動データ: {len(activity_data)}件')
print(f'Fitbitデータ: {len(fitbit_data)}件')
"

# 2. モデル訓練実行
python -c "
from ml_model import FrustrationPredictor

predictor = FrustrationPredictor()
performance = predictor.train_initial_model('default')
print('初期モデル訓練完了:', performance)
"
```

#### 2. 継続的なモデル改善

```python
# 定期的なモデル再訓練（1日1回実行推奨）
def daily_model_retrain():
    """日次モデル再訓練"""
    
    predictor = FrustrationPredictor()
    
    # 過去7日間のデータを取得
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    activity_data = sheets_connector.get_activity_data_range('default', start_date, end_date)
    fitbit_data = sheets_connector.get_fitbit_data_range('default', start_date, end_date)
    
    # モデル更新
    predictor.update_model_with_new_data(activity_data, fitbit_data)
    
    # 性能評価
    performance = predictor.evaluate_recent_performance()
    
    logger.info(f"日次モデル更新完了: {performance}")
    
    return performance
```

#### 3. モデル性能監視

```python
def monitor_model_performance():
    """モデル性能の継続監視"""
    
    # 直近の予測精度をチェック
    recent_predictions = sheets_connector.get_recent_predictions(days=3)
    actual_values = sheets_connector.get_actual_frustration_values(days=3)
    
    if len(recent_predictions) > 0 and len(actual_values) > 0:
        mae = mean_absolute_error(actual_values, recent_predictions)
        
        # 性能劣化の閾値チェック
        if mae > 5.0:  # MAEが5.0を超えた場合
            logger.warning(f"モデル性能劣化検出: MAE={mae}")
            
            # 自動再訓練をトリガー
            daily_model_retrain()
            
            return False
    
    return True
```

### 特徴量重要度分析

```python
def analyze_feature_importance():
    """特徴量重要度の分析"""
    
    if hasattr(self.current_model, 'feature_importances_'):
        importance_dict = dict(zip(
            self.feature_columns, 
            self.current_model.feature_importances_
        ))
        
        # 重要度順でソート
        sorted_importance = sorted(
            importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        print("特徴量重要度 (上位10):")
        for feature, importance in sorted_importance[:10]:
            print(f"  {feature}: {importance:.4f}")
        
        return sorted_importance
    
    return []
```

### モデルのバックアップ・復元

```python
def save_model_backup():
    """モデルのバックアップ保存"""
    import joblib
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"models/frustration_model_backup_{timestamp}.pkl"
    
    joblib.dump(self.current_model, backup_path)
    logger.info(f"モデルバックアップ保存: {backup_path}")
    
    return backup_path

def restore_model_from_backup(backup_path):
    """バックアップからモデルを復元"""
    import joblib
    
    self.current_model = joblib.load(backup_path)
    logger.info(f"モデル復元完了: {backup_path}")
```

## 🎯 今後の拡張予定

- **複数ユーザー対応**: チーム・組織での利用
- **予測精度向上**: より高度な時系列モデルの導入
- **リアルタイム連携**: Fitbitリアルタイムデータ取得
- **モバイルアプリ**: プッシュ通知対応
- **カスタマイズ**: ユーザー別の重みづけ・閾値設定

## 📄 ライセンス

MIT License

---

## 🆘 サポート

システムに関する質問や問題は、GitHub Issues または開発者にお問い合わせください。