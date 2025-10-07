"""
機械学習モデルのデバッグ用スクリプト
"""

import pandas as pd
import numpy as np
from datetime import datetime
from ml_model import FrustrationPredictor
from sheets_connector import SheetsConnector
from config import Config
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_model():
    """モデルの状態を詳細にデバッグ"""

    print("=" * 80)
    print("機械学習モデル デバッグレポート")
    print("=" * 80)
    print()

    # 初期化
    predictor = FrustrationPredictor()
    sheets_connector = SheetsConnector()
    config = Config()

    # ユーザーID
    user_id = 'default'

    # データ取得
    print("📊 データ取得中...")
    activity_data = sheets_connector.get_activity_data(user_id)
    fitbit_data = sheets_connector.get_fitbit_data(user_id)

    print(f"  - 活動データ: {len(activity_data)} 件")
    print(f"  - Fitbitデータ: {len(fitbit_data)} 件")
    print()

    if activity_data.empty:
        print("❌ エラー: 活動データがありません")
        return

    # データ前処理
    print("🔧 データ前処理...")
    activity_processed = predictor.preprocess_activity_data(activity_data)
    df_enhanced = predictor.aggregate_fitbit_by_activity(activity_processed, fitbit_data)

    print(f"  - 前処理後: {len(df_enhanced)} 件")
    print()

    # データ品質チェック
    print("📈 データ品質チェック...")
    data_quality = predictor.check_data_quality(df_enhanced)
    print(f"  - 総サンプル数: {data_quality['total_samples']}")
    print(f"  - データ充足: {data_quality['is_sufficient']}")
    print(f"  - 品質レベル: {data_quality['quality_level']}")
    if data_quality['warnings']:
        print(f"  ⚠️ 警告:")
        for warning in data_quality['warnings']:
            print(f"    - {warning}")
    print()

    # NASA_F統計
    print("📊 NASA_F統計...")
    if 'NASA_F' in df_enhanced.columns:
        nasa_f = df_enhanced['NASA_F'].dropna()
        print(f"  - 件数: {len(nasa_f)}")
        print(f"  - 平均: {nasa_f.mean():.2f}")
        print(f"  - 標準偏差: {nasa_f.std():.2f}")
        print(f"  - 最小値: {nasa_f.min():.2f}")
        print(f"  - 最大値: {nasa_f.max():.2f}")
        print(f"  - ユニーク値: {nasa_f.nunique()}")
        print(f"  - 値の分布: {nasa_f.value_counts().head(10).to_dict()}")
    print()

    # 活動カテゴリ統計
    print("🏃 活動カテゴリ統計...")
    if 'CatSub' in df_enhanced.columns:
        activities = df_enhanced['CatSub'].dropna()
        print(f"  - 総活動数: {len(activities)}")
        print(f"  - ユニーク活動数: {activities.nunique()}")
        print(f"  - 上位5活動:")
        for activity, count in activities.value_counts().head(5).items():
            print(f"    - {activity}: {count}件")
    print()

    # モデル訓練
    print("🤖 モデル訓練...")
    if len(df_enhanced) >= 10:
        training_results = predictor.walk_forward_validation_train(df_enhanced)
        print(f"  - モデル訓練: ✅ 完了")
        print(f"  - RMSE: {training_results.get('walk_forward_rmse', 'N/A'):.2f}")
        print(f"  - MAE: {training_results.get('walk_forward_mae', 'N/A'):.2f}")
        print(f"  - R²: {training_results.get('walk_forward_r2', 'N/A'):.3f}")

        # 予測値の多様性
        pred_diversity = training_results.get('prediction_diversity', {})
        print(f"  - 予測値標準偏差: {pred_diversity.get('std', 0):.3f}")
        print(f"  - 予測値種類数: {pred_diversity.get('unique_values', 0)}")
        print(f"  - 多様性OK: {pred_diversity.get('is_diverse', False)}")

        # 特徴量重要度
        print(f"\n  📊 特徴量重要度（上位10）:")
        feature_importance = training_results.get('feature_importance', {})
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        for feature, importance in sorted_features:
            print(f"    - {feature}: {importance:.4f}")
    else:
        print(f"  ❌ データ不足: {len(df_enhanced)}件 < 10件")
    print()

    # エンコーダー確認
    print("🔑 エンコーダー確認...")
    if hasattr(predictor, 'encoders'):
        for key, encoder in predictor.encoders.items():
            print(f"  - {key}:")
            if hasattr(encoder, 'classes_'):
                print(f"    クラス数: {len(encoder.classes_)}")
                print(f"    クラス: {list(encoder.classes_[:5])}...")
    print()

    # テスト予測（過去データから）
    print("🔮 テスト予測...")
    if not df_enhanced.empty and predictor.model is not None:
        test_cases = df_enhanced.head(5)

        for idx, row in test_cases.iterrows():
            activity = row.get('CatSub', 'unknown')
            duration = row.get('Duration', 60)
            timestamp = pd.to_datetime(row.get('Timestamp'))
            actual_frustration = row.get('NASA_F')

            print(f"\n  テストケース {idx + 1}:")
            print(f"    - 活動: {activity}")
            print(f"    - 時刻: {timestamp}")
            print(f"    - 実測値: {actual_frustration}")

            # predict_single_activity（固定値使用）
            result1 = predictor.predict_single_activity(activity, duration, timestamp)
            print(f"    - 予測値(固定特徴量): {result1.get('predicted_frustration', 'N/A'):.2f}")

            # predict_with_history（履歴使用）
            result2 = predictor.predict_with_history(activity, duration, timestamp, df_enhanced)
            print(f"    - 予測値(履歴使用): {result2.get('predicted_frustration', 'N/A'):.2f}")
            print(f"    - 使用履歴数: {result2.get('historical_records', 'N/A')}")
    print()

    # 最新データでの予測テスト
    print("🎯 最新データでの予測テスト...")
    if not df_enhanced.empty and predictor.model is not None:
        latest = df_enhanced.iloc[-1]
        activity = latest.get('CatSub', 'unknown')
        duration = latest.get('Duration', 60)
        timestamp = datetime.now()

        print(f"  - 活動: {activity}")
        print(f"  - 期間: {duration}分")

        result = predictor.predict_with_history(activity, duration, timestamp, df_enhanced)
        print(f"  - 予測値: {result.get('predicted_frustration', 'N/A'):.2f}")
        print(f"  - 信頼度: {result.get('confidence', 0):.3f}")
        print(f"  - 使用履歴数: {result.get('historical_records', 'N/A')}")
    print()

    # 診断結果
    print("=" * 80)
    print("🔍 診断結果")
    print("=" * 80)

    issues = []

    if data_quality['total_samples'] < 10:
        issues.append(f"データ数不足: {data_quality['total_samples']}件 < 10件")

    if 'NASA_F' in df_enhanced.columns:
        nasa_f = df_enhanced['NASA_F'].dropna()
        if nasa_f.std() < 1.0:
            issues.append(f"NASA_Fの分散が小さい: σ={nasa_f.std():.2f}")
        if nasa_f.nunique() < 3:
            issues.append(f"NASA_Fの種類が少ない: {nasa_f.nunique()}種類")

    if predictor.model is None:
        issues.append("モデルが訓練されていません")

    if 'CatSub' in df_enhanced.columns:
        if df_enhanced['CatSub'].nunique() < 3:
            issues.append(f"活動の種類が少ない: {df_enhanced['CatSub'].nunique()}種類")

    if issues:
        print("❌ 検出された問題:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ 問題なし: モデルは正常に動作しています")

    print()
    print("=" * 80)

if __name__ == '__main__':
    debug_model()
