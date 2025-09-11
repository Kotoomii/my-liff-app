"""
フラストレーション値予測・反実仮想説明システム
行動変更タイミングでの予測とDiCEによる改善提案を提供するFlaskアプリケーション
"""

from flask import Flask, render_template, jsonify, request
import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict

from ml_model import FrustrationPredictor
from sheets_connector import SheetsConnector
from counterfactual_explainer import ActivityCounterfactualExplainer
from llm_feedback_generator import LLMFeedbackGenerator
from scheduler import FeedbackScheduler

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# グローバルインスタンス
predictor = FrustrationPredictor()
sheets_connector = SheetsConnector()
explainer = ActivityCounterfactualExplainer()
feedback_generator = LLMFeedbackGenerator()
scheduler = FeedbackScheduler()

@app.route('/')
def index():
    """メインダッシュボード - 過去24時間のDiCE結果可視化"""
    import time
    timestamp = str(int(time.time()))
    return render_template('frustration_dashboard.html', timestamp=timestamp)

@app.route('/mirror')
def smart_mirror():
    """スマートミラー専用UI - 完全自動運転・タッチレス操作"""
    return render_template('smart_mirror.html')

@app.route('/tablet')
def tablet_mirror():
    """タブレット専用UI - 手動ユーザー選択・日次平均表示"""
    return render_template('tablet_mirror.html')

@app.route('/trends')
def frustration_trends():
    """ユーザー別フラストレーション値推移確認シート"""
    return render_template('frustration_trends.html')

@app.route('/api/frustration/predict', methods=['POST'])
def predict_frustration():
    """
    行動変更タイミングでのフラストレーション値予測API
    """
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default')
        target_timestamp = data.get('timestamp')
        
        if target_timestamp:
            target_timestamp = datetime.fromisoformat(target_timestamp)
        
        # データ取得
        activity_data = sheets_connector.get_activity_data(user_id)
        fitbit_data = sheets_connector.get_fitbit_data(user_id)
        
        if activity_data.empty:
            return jsonify({
                'status': 'error',
                'message': '活動データが見つかりません'
            }), 400
        
        # データ前処理
        activity_processed = predictor.preprocess_activity_data(activity_data)
        if activity_processed.empty:
            return jsonify({
                'status': 'error', 
                'message': 'データの前処理に失敗しました'
            }), 400
        
        # Fitbitデータとの統合
        df_enhanced = predictor.aggregate_fitbit_by_activity(activity_processed, fitbit_data)
        
        # Walk Forward Validationで学習
        if len(df_enhanced) > 10:
            training_results = predictor.walk_forward_validation_train(df_enhanced)
        else:
            training_results = {}
        
        # フラストレーション値予測
        prediction_result = predictor.predict_frustration_at_activity_change(
            df_enhanced, target_timestamp
        )
        
        # 行動変更タイミング一覧
        change_timestamps = predictor.get_activity_change_timestamps(df_enhanced)
        
        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'prediction': prediction_result,
            'activity_change_timestamps': [ts.isoformat() for ts in change_timestamps],
            'model_performance': training_results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"フラストレーション値予測エラー: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/frustration/dice-analysis', methods=['POST'])
def generate_dice_analysis():
    """
    過去24時間の行動に対するDiCE分析API
    """
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default')
        target_timestamp = data.get('timestamp')
        lookback_hours = data.get('lookback_hours', 24)
        
        if target_timestamp:
            target_timestamp = datetime.fromisoformat(target_timestamp)
        
        # データ取得
        activity_data = sheets_connector.get_activity_data(user_id)
        fitbit_data = sheets_connector.get_fitbit_data(user_id)
        
        if activity_data.empty:
            return jsonify({
                'status': 'error',
                'message': '活動データが見つかりません'
            }), 400
        
        # データ前処理
        activity_processed = predictor.preprocess_activity_data(activity_data)
        if activity_processed.empty:
            return jsonify({
                'status': 'error',
                'message': 'データの前処理に失敗しました'
            }), 400
        
        # Fitbitデータとの統合
        df_enhanced = predictor.aggregate_fitbit_by_activity(activity_processed, fitbit_data)
        
        # モデル学習（必要に応じて）
        if len(df_enhanced) > 10 and predictor.model is None:
            predictor.walk_forward_validation_train(df_enhanced)
        
        # DiCE分析実行
        dice_result = explainer.generate_activity_based_explanation(
            df_enhanced, predictor, target_timestamp, lookback_hours
        )
        
        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'dice_analysis': dice_result,
            'lookback_hours': lookback_hours,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"DiCE分析エラー: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/frustration/timeline', methods=['POST'])
def get_frustration_timeline():
    """
    過去24時間のフラストレーション値タイムライン取得API
    """
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default')
        date = data.get('date', datetime.now().strftime('%Y-%m-%d'))
        
        # データ取得
        activity_data = sheets_connector.get_activity_data(user_id)
        fitbit_data = sheets_connector.get_fitbit_data(user_id)
        
        if activity_data.empty:
            return jsonify({
                'status': 'error',
                'message': '活動データが見つかりません'
            }), 400
        
        # 指定日のデータをフィルタリング
        target_date = datetime.strptime(date, '%Y-%m-%d').date()
        if 'Timestamp' in activity_data.columns:
            activity_data['date'] = pd.to_datetime(activity_data['Timestamp']).dt.date
            daily_data = activity_data[activity_data['date'] == target_date]
        else:
            daily_data = pd.DataFrame()
        
        if daily_data.empty:
            return jsonify({
                'status': 'success',
                'date': date,
                'timeline': [],
                'message': '指定日のデータが見つかりません'
            })
        
        # データ前処理
        activity_processed = predictor.preprocess_activity_data(daily_data)
        df_enhanced = predictor.aggregate_fitbit_by_activity(activity_processed, fitbit_data)
        
        # タイムライン作成
        timeline = []
        for idx, row in df_enhanced.iterrows():
            timeline.append({
                'timestamp': row['Timestamp'].isoformat(),
                'hour': row.get('hour', 0),
                'activity': row.get('CatSub', 'unknown'),
                'duration': row.get('Duration', 0),
                'frustration_value': row.get('NASA_F', 10),
                'activity_change': row.get('activity_change', 0) == 1,
                'lorenz_stats': {
                    'mean': row.get('lorenz_mean', 0),
                    'std': row.get('lorenz_std', 0)
                }
            })
        
        # 時間順にソート
        timeline.sort(key=lambda x: x['timestamp'])
        
        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'date': date,
            'timeline': timeline,
            'total_entries': len(timeline)
        })
        
    except Exception as e:
        logger.error(f"タイムライン取得エラー: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/feedback/generate', methods=['POST'])
def generate_feedback():
    """
    LLMを使用した自然言語フィードバック生成API
    """
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default')
        feedback_type = data.get('type', 'evening')  # 'morning' or 'evening'
        
        # DiCE結果を取得（過去24時間）
        dice_results = []
        
        # 過去24時間のDiCE分析を実行
        for hours_back in range(0, 24, 6):  # 6時間おきに分析
            target_time = datetime.now() - timedelta(hours=hours_back)
            
            # データ取得
            activity_data = sheets_connector.get_activity_data(user_id)
            fitbit_data = sheets_connector.get_fitbit_data(user_id)
            
            if not activity_data.empty:
                activity_processed = predictor.preprocess_activity_data(activity_data)
                df_enhanced = predictor.aggregate_fitbit_by_activity(activity_processed, fitbit_data)
                
                if len(df_enhanced) > 5:
                    dice_result = explainer.generate_activity_based_explanation(
                        df_enhanced, predictor, target_time, 6
                    )
                    
                    if dice_result.get('type') != 'fallback':
                        dice_results.append(dice_result)
        
        # フィードバック生成
        if feedback_type == 'morning':
            feedback_result = feedback_generator.generate_morning_briefing(dice_results)
        else:
            feedback_result = feedback_generator.generate_evening_summary(dice_results)
        
        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'feedback_type': feedback_type,
            'feedback': feedback_result,
            'dice_results_count': len(dice_results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"フィードバック生成エラー: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/scheduler/status', methods=['GET'])
def scheduler_status():
    """定期フィードバックスケジューラーの状態取得"""
    try:
        status = scheduler.get_status()
        return jsonify({
            'status': 'success',
            'scheduler': status
        })
    except Exception as e:
        logger.error(f"スケジューラー状態取得エラー: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/scheduler/config', methods=['POST'])
def update_scheduler_config():
    """定期フィードバックスケジューラーの設定更新"""
    try:
        data = request.get_json()
        morning_time = data.get('morning_time')
        evening_time = data.get('evening_time')
        enabled = data.get('enabled')
        
        scheduler.update_schedule_config(morning_time, evening_time, enabled)
        
        return jsonify({
            'status': 'success',
            'message': 'スケジューラー設定を更新しました'
        })
    except Exception as e:
        logger.error(f"スケジューラー設定更新エラー: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/scheduler/trigger', methods=['POST'])
def trigger_manual_feedback():
    """手動フィードバック実行"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default')
        feedback_type = data.get('type', 'evening')
        
        feedback = scheduler.trigger_manual_feedback(user_id, feedback_type)
        
        return jsonify({
            'status': 'success',
            'feedback': feedback,
            'message': f'{feedback_type}フィードバックを手動実行しました'
        })
    except Exception as e:
        logger.error(f"手動フィードバック実行エラー: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/feedback/history', methods=['GET'])
def get_feedback_history():
    """フィードバック履歴取得"""
    try:
        user_id = request.args.get('user_id')
        days = int(request.args.get('days', 7))
        
        history = scheduler.get_recent_feedback(user_id, days)
        
        return jsonify({
            'status': 'success',
            'history': history,
            'count': len(history)
        })
    except Exception as e:
        logger.error(f"フィードバック履歴取得エラー: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """ヘルスチェック"""
    try:
        # 各コンポーネントの状態確認
        sheets_status = sheets_connector.test_connection()
        scheduler_status = scheduler.get_status()
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'sheets_connector': sheets_status,
                'scheduler': scheduler_status,
                'predictor_ready': predictor.model is not None,
                'explainer_ready': explainer is not None,
                'feedback_generator_ready': feedback_generator is not None
            }
        })
    except Exception as e:
        logger.error(f"ヘルスチェックエラー: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/users', methods=['GET'])
def get_users():
    """利用可能ユーザー一覧取得"""
    try:
        # 複数ユーザー対応：LINEユーザーIDとFitbitシートの組み合わせ設定
        users = [
            {
                'user_id': 'default', 
                'name': 'デフォルトユーザー', 
                'icon': '👤',
                'activity_sheet': 'default',  # LINE活動報告シート名
                'fitbit_sheet': 'kotoomi_Fitbit-data-default'  # Fitbitデータシート名
            },
            {
                'user_id': 'user1', 
                'name': 'ユーザー1', 
                'icon': '👨',
                'activity_sheet': 'U1234567890abcdef',  # LINEユーザーID例
                'fitbit_sheet': 'kotoomi_Fitbit-data-01'
            },
            {
                'user_id': 'user2', 
                'name': 'ユーザー2', 
                'icon': '👩',
                'activity_sheet': 'U2345678901bcdefg',  # LINEユーザーID例
                'fitbit_sheet': 'kotoomi_Fitbit-data-02'
            },
            {
                'user_id': 'user3', 
                'name': 'ユーザー3', 
                'icon': '🧑',
                'activity_sheet': 'U3456789012cdefgh',  # LINEユーザーID例
                'fitbit_sheet': 'kotoomi_Fitbit-data-03'
            },
            # 追加ユーザー登録例
            # {
            #     'user_id': 'kotoomi', 
            #     'name': 'こときみ', 
            #     'icon': '👩‍🔬',
            #     'activity_sheet': 'Uabc123def456ghi',  # 実際のLINEユーザーID
            #     'fitbit_sheet': 'kotoomi_Fitbit-data-main'
            # },
        ]
        
        return jsonify({
            'status': 'success',
            'users': users
        })
    except Exception as e:
        logger.error(f"ユーザー一覧取得エラー: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/frustration/trends', methods=['POST'])
def get_frustration_trends():
    """ユーザー別フラストレーション値推移データ取得"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default')
        days = data.get('days', 30)  # デフォルト30日間
        
        # データ取得
        activity_data = sheets_connector.get_activity_data(user_id)
        
        if activity_data.empty:
            return jsonify({
                'status': 'error',
                'message': '活動データが見つかりません'
            }), 400
        
        # 日別平均フラストレーション値を計算
        activity_data['Date'] = pd.to_datetime(activity_data['Timestamp']).dt.date
        
        # 過去指定日数のデータをフィルタリング
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        activity_data = activity_data[activity_data['Date'] >= start_date]
        
        # 日別統計の計算
        daily_stats = []
        date_range = [start_date + timedelta(days=x) for x in range(days)]
        
        for date in date_range:
            day_data = activity_data[activity_data['Date'] == date]
            
            if not day_data.empty:
                avg_frustration = day_data['NASA_F'].mean()
                min_frustration = day_data['NASA_F'].min()
                max_frustration = day_data['NASA_F'].max()
                activity_count = len(day_data)
                total_duration = day_data['Duration'].sum()
                
                # 活動別集計
                activity_summary = day_data.groupby('CatSub').agg({
                    'NASA_F': ['mean', 'count'],
                    'Duration': 'sum'
                }).round(1)
                
                activity_breakdown = []
                for activity in activity_summary.index:
                    activity_breakdown.append({
                        'activity': activity,
                        'avg_frustration': activity_summary.loc[activity, ('NASA_F', 'mean')],
                        'count': int(activity_summary.loc[activity, ('NASA_F', 'count')]),
                        'total_duration': int(activity_summary.loc[activity, ('Duration', 'sum')])
                    })
            else:
                avg_frustration = None
                min_frustration = None
                max_frustration = None
                activity_count = 0
                total_duration = 0
                activity_breakdown = []
            
            daily_stats.append({
                'date': date.isoformat(),
                'avg_frustration': avg_frustration,
                'min_frustration': min_frustration,
                'max_frustration': max_frustration,
                'activity_count': activity_count,
                'total_duration': total_duration,
                'activity_breakdown': activity_breakdown
            })
        
        # 期間全体の統計
        if not activity_data.empty:
            period_stats = {
                'avg_frustration': activity_data['NASA_F'].mean(),
                'min_frustration': activity_data['NASA_F'].min(),
                'max_frustration': activity_data['NASA_F'].max(),
                'total_activities': len(activity_data),
                'total_duration': activity_data['Duration'].sum(),
                'improvement_trend': calculate_trend(activity_data)
            }
        else:
            period_stats = {
                'avg_frustration': 0,
                'min_frustration': 0,
                'max_frustration': 0,
                'total_activities': 0,
                'total_duration': 0,
                'improvement_trend': 0
            }
        
        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'period': f'{days}日間',
            'daily_trends': daily_stats,
            'period_summary': period_stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"推移データ取得エラー: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def calculate_trend(activity_data):
    """フラストレーション値の改善トレンドを計算"""
    try:
        if len(activity_data) < 2:
            return 0
        
        # 日付順にソート
        activity_data = activity_data.sort_values('Timestamp')
        
        # 前半と後半に分けて平均値を比較
        mid_point = len(activity_data) // 2
        first_half_avg = activity_data.iloc[:mid_point]['NASA_F'].mean()
        second_half_avg = activity_data.iloc[mid_point:]['NASA_F'].mean()
        
        # 改善度を計算（負の値が改善を意味する）
        trend = second_half_avg - first_half_avg
        return round(trend, 2)
        
    except Exception:
        return 0

def get_user_config(user_id: str) -> Dict:
    """ユーザー設定を取得"""
    # main.pyのusers配列と同じ設定を取得
    users_config = [
        {
            'user_id': 'default', 
            'name': 'デフォルトユーザー', 
            'icon': '👤',
            'activity_sheet': 'default',
            'fitbit_sheet': 'kotoomi_Fitbit-data-default'
        },
        {
            'user_id': 'user1', 
            'name': 'ユーザー1', 
            'icon': '👨',
            'activity_sheet': 'U1234567890abcdef',
            'fitbit_sheet': 'kotoomi_Fitbit-data-01'
        },
        {
            'user_id': 'user2', 
            'name': 'ユーザー2', 
            'icon': '👩',
            'activity_sheet': 'U2345678901bcdefg',
            'fitbit_sheet': 'kotoomi_Fitbit-data-02'
        },
        {
            'user_id': 'user3', 
            'name': 'ユーザー3', 
            'icon': '🧑',
            'activity_sheet': 'U3456789012cdefgh',
            'fitbit_sheet': 'kotoomi_Fitbit-data-03'
        },
    ]
    
    for user in users_config:
        if user['user_id'] == user_id:
            return user
    
    # デフォルトを返す
    return users_config[0]

def initialize_application():
    """アプリケーション初期化"""
    try:
        logger.info("アプリケーションを初期化しています...")
        
        # スケジューラー開始
        scheduler.start_scheduler()
        logger.info("定期フィードバックスケジューラーを開始しました")
        
        logger.info("アプリケーション初期化完了")
    except Exception as e:
        logger.error(f"アプリケーション初期化エラー: {e}")

def cleanup_application():
    """アプリケーション終了処理"""
    try:
        logger.info("アプリケーションを終了しています...")
        scheduler.stop_scheduler()
        logger.info("アプリケーション終了完了")
    except Exception as e:
        logger.error(f"アプリケーション終了エラー: {e}")

if __name__ == '__main__':
    try:
        # アプリケーション初期化
        initialize_application()
        
        # 開発サーバー起動
        port = int(os.environ.get('PORT', 8080))
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        logger.info("キーボード割り込みによる終了")
    except Exception as e:
        logger.error(f"アプリケーション実行エラー: {e}")
    finally:
        cleanup_application()