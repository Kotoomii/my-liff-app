"""
複数ユーザー対応反実仮想機械学習システムのテストスクリプト
config.py、sheets_connector.py、ml_model.py、counterfactual_explainer.pyの統合動作を確認
"""

import pandas as pd
import logging
from datetime import datetime, timedelta
import sys
import traceback

from config import Config
from sheets_connector import SheetsConnector
from ml_model import FrustrationPredictor
from counterfactual_explainer import ActivityCounterfactualExplainer

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_user_data_retrieval():
    """ユーザー別データ取得テスト"""
    print("\n=== ユーザー別データ取得テスト ===")
    
    config = Config()
    connector = SheetsConnector()
    
    # 全ユーザーの設定を表示
    print("\n📋 設定済みユーザー一覧:")
    all_users = config.get_all_users()
    for user in all_users:
        print(f"  👤 {user['user_id']}: {user['name']} ({user['description']})")
        print(f"      📊 活動シート: {user['activity_sheet']}")
        print(f"      💓 Fitbitシート: {user['fitbit_sheet']}")
    
    # 各ユーザーのデータ取得テスト
    test_results = {}
    
    for user in all_users:
        user_id = user['user_id']
        print(f"\n🔍 {user_id} のデータ取得中...")
        
        try:
            # 活動データ取得
            activity_data = connector.get_activity_data(user_id)
            fitbit_data = connector.get_fitbit_data(user_id)
            
            result = {
                'user_id': user_id,
                'activity_data_count': len(activity_data),
                'fitbit_data_count': len(fitbit_data),
                'activity_data_empty': activity_data.empty,
                'fitbit_data_empty': fitbit_data.empty,
                'status': 'success' if not (activity_data.empty and fitbit_data.empty) else 'no_data'
            }
            
            if not activity_data.empty:
                print(f"  ✅ 活動データ: {len(activity_data)} 行")
                print(f"     📅 期間: {activity_data['Timestamp'].min()} - {activity_data['Timestamp'].max()}")
                if 'NASA_F' in activity_data.columns:
                    print(f"     😤 フラストレーション範囲: {activity_data['NASA_F'].min():.1f} - {activity_data['NASA_F'].max():.1f}")
            else:
                print(f"  ❌ 活動データ: データなし")
            
            if not fitbit_data.empty:
                print(f"  ✅ Fitbitデータ: {len(fitbit_data)} 行")
                print(f"     📅 期間: {fitbit_data['Timestamp'].min()} - {fitbit_data['Timestamp'].max()}")
            else:
                print(f"  ❌ Fitbitデータ: データなし")
                
            test_results[user_id] = result
            
        except Exception as e:
            print(f"  ❌ エラー: {e}")
            test_results[user_id] = {
                'user_id': user_id,
                'status': 'error',
                'error': str(e)
            }
    
    return test_results

def test_ml_model_training(user_id: str = 'default'):
    """機械学習モデル訓練テスト"""
    print(f"\n=== 機械学習モデル訓練テスト ({user_id}) ===")
    
    try:
        connector = SheetsConnector()
        predictor = FrustrationPredictor()
        
        # データ取得
        print(f"📊 {user_id} のデータを取得中...")
        activity_data = connector.get_activity_data(user_id)
        fitbit_data = connector.get_fitbit_data(user_id)
        
        if activity_data.empty:
            print("❌ 活動データが空のため、機械学習テストをスキップします")
            return None
        
        print(f"✅ 活動データ: {len(activity_data)} 行")
        print(f"✅ Fitbitデータ: {len(fitbit_data)} 行")
        
        # データ前処理
        print("🔄 データ前処理中...")
        processed_activity = predictor.preprocess_activity_data(activity_data)
        if processed_activity.empty:
            print("❌ データ前処理後にデータが空になりました")
            return None
        
        print(f"✅ 前処理完了: {len(processed_activity)} 行")
        
        # Fitbitデータ統合
        print("🔄 Fitbitデータ統合中...")
        enhanced_data = predictor.aggregate_fitbit_by_activity(processed_activity, fitbit_data)
        
        if enhanced_data.empty:
            print("❌ Fitbitデータ統合後にデータが空になりました")
            return None
        
        print(f"✅ データ統合完了: {len(enhanced_data)} 行")
        
        # Walk Forward Validation実行
        print("🤖 Walk Forward Validation実行中...")
        training_results = predictor.walk_forward_validation_train(enhanced_data)
        
        if training_results:
            print("✅ 機械学習訓練完了!")
            print(f"   📊 RMSE: {training_results.get('walk_forward_rmse', 'N/A'):.2f}")
            print(f"   📊 MAE: {training_results.get('walk_forward_mae', 'N/A'):.2f}")
            print(f"   📊 R²: {training_results.get('walk_forward_r2', 'N/A'):.3f}")
            print(f"   🔮 予測数: {training_results.get('total_predictions', 'N/A')}")
            
            # 特徴量重要度表示（上位5つ）
            feature_importance = training_results.get('feature_importance', {})
            if feature_importance:
                print("   🎯 特徴量重要度 (上位5つ):")
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                for feature, importance in sorted_features:
                    print(f"      {feature}: {importance:.3f}")
            
            return {
                'status': 'success',
                'results': training_results,
                'predictor': predictor,
                'enhanced_data': enhanced_data
            }
        else:
            print("❌ 機械学習訓練に失敗しました")
            return None
            
    except Exception as e:
        print(f"❌ 機械学習テストエラー: {e}")
        traceback.print_exc()
        return None

def test_counterfactual_explanation(predictor, enhanced_data, user_id: str = 'default'):
    """反実仮想説明テスト"""
    print(f"\n=== 反実仮想説明テスト ({user_id}) ===")
    
    try:
        explainer = ActivityCounterfactualExplainer()
        
        # 反実仮想説明生成
        print("🔮 反実仮想説明生成中...")
        explanation_result = explainer.generate_activity_based_explanation(
            enhanced_data, 
            predictor,
            target_timestamp=datetime.now(),
            lookback_hours=24
        )
        
        if explanation_result:
            print("✅ 反実仮想説明生成完了!")
            print(f"   📈 改善可能性: {explanation_result.get('total_improvement', 0):.1f}点")
            print(f"   📊 提案数: {explanation_result.get('num_suggestions', 0)}")
            print(f"   🎯 信頼度: {explanation_result.get('confidence', 0):.2f}")
            
            # トップ提案を表示
            top_suggestions = explanation_result.get('top_suggestions', [])
            if top_suggestions:
                print("   💡 主要な提案:")
                for i, suggestion in enumerate(top_suggestions[:3], 1):
                    print(f"      {i}. {suggestion}")
            
            # タイムライン表示
            timeline = explanation_result.get('timeline', [])
            if timeline:
                print(f"   ⏰ タイムライン結果: {len(timeline)} 件")
                for item in timeline[:3]:  # 最初の3件のみ表示
                    print(f"      {item['timestamp'].strftime('%H:%M')}: "
                          f"{item['original_activity']} → {item['suggested_activity']} "
                          f"(改善: {item['frustration_reduction']:.1f}点)")
            
            return {
                'status': 'success',
                'explanation': explanation_result
            }
        else:
            print("❌ 反実仮想説明生成に失敗しました")
            return None
            
    except Exception as e:
        print(f"❌ 反実仮想説明テストエラー: {e}")
        traceback.print_exc()
        return None

def test_prediction_for_user(predictor, enhanced_data, user_id: str = 'default'):
    """ユーザー別予測テスト"""
    print(f"\n=== フラストレーション予測テスト ({user_id}) ===")
    
    try:
        # 最新の行動変更タイミングで予測
        change_timestamps = predictor.get_activity_change_timestamps(enhanced_data, hours_back=24)
        
        if not change_timestamps:
            print("❌ 行動変更タイミングが見つかりません")
            return None
        
        print(f"📅 過去24時間の行動変更: {len(change_timestamps)} 回")
        
        # 最新のタイミングで予測
        latest_timestamp = max(change_timestamps)
        print(f"🔮 予測対象時刻: {latest_timestamp}")
        
        prediction_result = predictor.predict_frustration_at_activity_change(
            enhanced_data, 
            target_timestamp=latest_timestamp
        )
        
        if prediction_result:
            print("✅ 予測完了!")
            print(f"   🎯 予測フラストレーション値: {prediction_result.get('predicted_frustration', 'N/A'):.1f}")
            if prediction_result.get('actual_frustration') is not None:
                actual = prediction_result.get('actual_frustration')
                predicted = prediction_result.get('predicted_frustration')
                print(f"   📊 実際値: {actual:.1f}")
                print(f"   📏 誤差: {abs(actual - predicted):.1f}")
            print(f"   🏃 活動: {prediction_result.get('activity', 'N/A')}")
            print(f"   ⏱️ 所要時間: {prediction_result.get('duration', 'N/A')} 分")
            print(f"   🎯 信頼度: {prediction_result.get('confidence', 0):.2f}")
            
            return {
                'status': 'success',
                'prediction': prediction_result
            }
        else:
            print("❌ 予測に失敗しました")
            return None
            
    except Exception as e:
        print(f"❌ 予測テストエラー: {e}")
        traceback.print_exc()
        return None

def run_full_test():
    """全体テスト実行"""
    print("🚀 複数ユーザー対応反実仮想機械学習システム テスト開始")
    print("=" * 60)
    
    # 1. データ取得テスト
    data_results = test_user_data_retrieval()
    
    # 2. データが存在するユーザーで機械学習テスト
    successful_users = []
    for user_id, result in data_results.items():
        if result.get('status') == 'success' and not result.get('activity_data_empty', True):
            successful_users.append(user_id)
    
    if not successful_users:
        print("\n❌ 実際のデータが存在するユーザーが見つかりません")
        print("Google Sheetsの認証とデータを確認してください")
        return
    
    print(f"\n✅ データが存在するユーザー: {successful_users}")
    
    # 最初のユーザーで詳細テスト実行
    test_user = successful_users[0]
    print(f"\n🎯 詳細テスト対象ユーザー: {test_user}")
    
    # 3. 機械学習モデル訓練テスト
    ml_result = test_ml_model_training(test_user)
    
    if ml_result and ml_result.get('status') == 'success':
        predictor = ml_result['predictor']
        enhanced_data = ml_result['enhanced_data']
        
        # 4. 予測テスト
        pred_result = test_prediction_for_user(predictor, enhanced_data, test_user)
        
        # 5. 反実仮想説明テスト
        cf_result = test_counterfactual_explanation(predictor, enhanced_data, test_user)
        
        # 6. 結果サマリー
        print(f"\n🎉 テスト完了サマリー ({test_user})")
        print("=" * 40)
        print(f"✅ データ取得: 成功")
        print(f"✅ 機械学習訓練: 成功")
        print(f"✅ フラストレーション予測: {'成功' if pred_result else '失敗'}")
        print(f"✅ 反実仮想説明: {'成功' if cf_result else '失敗'}")
        
        return {
            'data_results': data_results,
            'ml_result': ml_result,
            'prediction_result': pred_result,
            'counterfactual_result': cf_result
        }
    else:
        print(f"\n❌ {test_user} の機械学習訓練に失敗したため、後続テストをスキップします")
        return None

if __name__ == "__main__":
    try:
        results = run_full_test()
        
        if results:
            print("\n🎯 統合テスト結果:")
            print("複数ユーザー対応反実仮想機械学習システムの動作確認が完了しました!")
            print("\n次のステップ:")
            print("1. ユーザー選択UIの実装")
            print("2. 各UIでの複数ユーザー対応")
            print("3. 本格運用のためのデータ蓄積")
        else:
            print("\n❌ 統合テストに失敗しました")
            print("設定とデータを確認してください")
            
    except Exception as e:
        print(f"\n💥 テスト実行中にエラーが発生しました: {e}")
        traceback.print_exc()