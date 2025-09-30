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
import threading
import time
import json

from ml_model import FrustrationPredictor
from sheets_connector import SheetsConnector
from counterfactual_explainer import ActivityCounterfactualExplainer
from llm_feedback_generator import LLMFeedbackGenerator
from scheduler import FeedbackScheduler
from config import Config

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# グローバルインスタンス
predictor = FrustrationPredictor()
sheets_connector = SheetsConnector()
explainer = ActivityCounterfactualExplainer()
feedback_generator = LLMFeedbackGenerator()
scheduler = FeedbackScheduler()

# DiCE daily scheduler
dice_scheduler_thread = None
dice_scheduler_running = False
last_dice_result = {}

# データ更新監視スレッド
data_monitor_thread = None
data_monitor_running = False
last_prediction_result = {}

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

@app.route('/activity')
def activity_page():
    """活動データ入力ページ (my-liff-app互換) - リダイレクト"""
    from flask import redirect
    return redirect('https://kotoomii.github.io/my-liff-app/activity.html')

@app.route('/nasa')
def nasa_page():
    """NASA-TLX評価ページ (my-liff-app互換) - リダイレクト"""
    from flask import redirect
    return redirect('https://kotoomii.github.io/my-liff-app/nasa.html')

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

@app.route('/api/frustration/predict-activity', methods=['POST'])
def predict_activity_frustration():
    """
    新しい活動入力時のリアルタイムフラストレーション予測API
    """
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default')
        activity_category = data.get('CatSub')  # 活動カテゴリ
        activity_subcategory = data.get('CatMid', activity_category)  # 活動サブカテゴリ
        duration = data.get('Duration', 60)  # 活動時間（分）
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        # 過去データ取得・前処理
        activity_data = sheets_connector.get_activity_data(user_id)
        fitbit_data = sheets_connector.get_fitbit_data(user_id)
        
        if activity_data.empty:
            # 初回利用時のデフォルト予測値を返す
            default_frustration = 10.0  # 1-20スケールの中間値
            return jsonify({
                'status': 'success',
                'user_id': user_id,
                'predicted_frustration': default_frustration,
                'activity': activity_category,
                'duration': duration,
                'confidence': 0.5,
                'message': '初回利用のためデフォルト値を返しました',
                'timestamp': timestamp.isoformat()
            })
        
        # データ前処理とモデル学習
        activity_processed = predictor.preprocess_activity_data(activity_data)
        df_enhanced = predictor.aggregate_fitbit_by_activity(activity_processed, fitbit_data)
        
        if len(df_enhanced) > 5:  # 最低5件のデータで学習
            predictor.walk_forward_validation_train(df_enhanced)
        
        # 新しい活動のフラストレーション値予測
        prediction_result = predictor.predict_single_activity(
            activity_category, 
            duration, 
            timestamp
        )
        
        if 'error' in prediction_result:
            return jsonify({
                'status': 'error',
                'message': prediction_result['error']
            }), 400
        
        predicted_frustration = prediction_result['predicted_frustration']
        confidence = prediction_result['confidence']
        
        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'predicted_frustration': round(predicted_frustration, 2),
            'activity': activity_category,
            'subcategory': activity_subcategory,
            'duration': duration,
            'confidence': round(confidence, 3),
            'timestamp': timestamp.isoformat()
        })
        
    except Exception as e:
        logger.error(f"活動フラストレーション予測エラー: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/frustration/daily-dice-schedule', methods=['POST'])
def generate_daily_dice_schedule():
    """
    1日の終わりに時間ごとのDiCE改善提案スケジュールを生成
    """
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default')
        target_date = data.get('date', datetime.now().date().isoformat())
        
        if isinstance(target_date, str):
            target_date = datetime.fromisoformat(target_date).date()
        
        # その日の活動データを取得
        activity_data = sheets_connector.get_activity_data(user_id)
        fitbit_data = sheets_connector.get_fitbit_data(user_id)
        
        if activity_data.empty:
            return jsonify({
                'status': 'error',
                'message': '活動データが見つかりません'
            }), 400
        
        # 指定日のデータをフィルタリング
        activity_data['Date'] = pd.to_datetime(activity_data['Timestamp']).dt.date
        daily_activities = activity_data[activity_data['Date'] == target_date].copy()
        
        if daily_activities.empty:
            return jsonify({
                'status': 'success',
                'user_id': user_id,
                'date': target_date.isoformat(),
                'message': 'その日の活動データがありません',
                'schedule': [],
                'recommendations': []
            })
        
        # データ前処理とモデル学習
        activity_processed = predictor.preprocess_activity_data(activity_data)
        df_enhanced = predictor.aggregate_fitbit_by_activity(activity_processed, fitbit_data)
        
        if len(df_enhanced) > 5:
            predictor.walk_forward_validation_train(df_enhanced)
        
        # 1時間ごとのスケジュール提案を生成（DiCE方式）
        dice_result = explainer.generate_hourly_alternatives(
            activity_data, 
            predictor,
            target_date
        )
        
        if dice_result and dice_result.get('hourly_schedule'):
            hourly_schedule = dice_result['hourly_schedule']
            message = dice_result.get('message', '時間別改善提案を生成しました')
            total_improvement = dice_result.get('total_improvement', 0)
        else:
            # フォールバック：基本的な時間別スケジュール
            hourly_schedule = []
            daily_activities = daily_activities.sort_values('Timestamp')
            
            for hour in range(24):  # 0-23時
                hour_start = datetime.combine(target_date, datetime.min.time()) + timedelta(hours=hour)
                hour_end = hour_start + timedelta(hours=1)
                
                # その時間帯の活動を取得
                hour_activities = daily_activities[
                    (daily_activities['Timestamp'] >= hour_start) & 
                    (daily_activities['Timestamp'] < hour_end)
                ]
                
                if not hour_activities.empty:
                    # 実際の活動とフラストレーション値
                    actual_activity = hour_activities.iloc[0]
                    actual_frustration = actual_activity.get('NASA_F', 10.0)
                    
                    # シンプルな提案（デフォルト）
                    suggested_activity = 'リラックス' if actual_activity['CatSub'] == '仕事' else '軽い運動'
                
                # その時間のスケジュール情報
                hour_info = {
                    'hour': hour,
                    'time_range': f"{hour:02d}:00-{(hour+1):02d}:00",
                    'actual_activity': actual_activity['CatSub'],
                    'actual_frustration': round(float(actual_frustration), 2),
                    'actual_duration': int(actual_activity.get('Duration', 60)),
                    'suggested_activity': None,
                    'suggested_frustration': None,
                    'improvement': 0.0,
                    'has_suggestion': False
                }
                
                # DiCE提案がある場合
                if dice_suggestions and dice_suggestions.get('alternatives'):
                    best_alternative = dice_suggestions['alternatives'][0]
                    suggested_frustration = best_alternative.get('predicted_frustration', actual_frustration)
                    improvement = actual_frustration - suggested_frustration
                    
                    if improvement > 0.5:  # 0.5以上の改善が見込める場合のみ提案
                        hour_info.update({
                            'suggested_activity': best_alternative.get('activity', actual_activity['CatSub']),
                            'suggested_frustration': round(suggested_frustration, 2),
                            'improvement': round(improvement, 2),
                            'has_suggestion': True
                        })
                        
                        # 日次推奨事項に追加
                        daily_recommendations.append({
                            'time': f"{hour:02d}:00",
                            'original': f"{actual_activity['CatSub']} (フラストレーション: {actual_frustration:.1f})",
                            'suggested': f"{best_alternative.get('activity')} (予測フラストレーション: {suggested_frustration:.1f})",
                            'improvement': round(improvement, 2),
                            'reason': f"この時間に{best_alternative.get('activity')}を行うことで、フラストレーションを{improvement:.1f}ポイント削減できます"
                        })
                
                hourly_schedule.append(hour_info)
            
            else:
                # その時間に活動がない場合
                hourly_schedule.append({
                    'hour': hour,
                    'time_range': f"{hour:02d}:00-{(hour+1):02d}:00",
                    'actual_activity': None,
                    'actual_frustration': None,
                    'actual_duration': 0,
                    'suggested_activity': None,
                    'suggested_frustration': None,
                    'improvement': 0.0,
                    'has_suggestion': False
                })
        
        # 全体統計
        total_actual_frustration = sum([h['actual_frustration'] for h in hourly_schedule if h['actual_frustration'] is not None])
        total_suggested_frustration = sum([h['suggested_frustration'] for h in hourly_schedule if h['suggested_frustration'] is not None])
        total_improvement = sum([h['improvement'] for h in hourly_schedule])
        
        # DiCEによる時間別改善提案レスポンス形式に統一
        if dice_result and dice_result.get('hourly_schedule'):
            return jsonify({
                'status': 'success',
                'user_id': user_id,
                'date': target_date.isoformat(),
                'schedule': dice_result['hourly_schedule'],
                'message': dice_result.get('message', '今日このような活動をしていたらストレスレベルが下がっていました'),
                'total_improvement': dice_result.get('total_improvement', 0),
                'summary': dice_result.get('summary', ''),
                'confidence': dice_result.get('confidence', 0.7),
                'generated_at': datetime.now().isoformat()
            })
        else:
            # フォールバック応答
            return jsonify({
                'status': 'success',
                'user_id': user_id,
                'date': target_date.isoformat(),
                'schedule': [],
                'message': '今日このような活動をしていたらストレスレベルが下がっていました',
                'total_improvement': 0,
                'summary': 'データが不足しているため、詳細な提案を生成できませんでした',
                'confidence': 0.3,
                'generated_at': datetime.now().isoformat()
            })
        
    except Exception as e:
        logger.error(f"日次DiCEスケジュール生成エラー: {e}")
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
        # リクエストデータの確認と修正
        if request.is_json:
            data = request.get_json()
            if data is None:
                data = {}
        else:
            data = {}
        
        user_id = data.get('user_id', 'default')
        date = data.get('date', datetime.now().strftime('%Y-%m-%d'))
        
        logger.info(f"Timeline API呼び出し - user_id: {user_id}, date: {date}")
        
        # データ取得
        activity_data = sheets_connector.get_activity_data(user_id)
        fitbit_data = sheets_connector.get_fitbit_data(user_id)
        
        # Google Sheets接続エラーの場合は適切なメッセージを返す
        if activity_data.empty:
            if sheets_connector.gc is None:
                return jsonify({
                    'status': 'success',
                    'date': date,
                    'timeline': [],
                    'message': 'Google Sheetsに接続できません。認証設定を確認してください。'
                })
            else:
                return jsonify({
                    'status': 'success', 
                    'date': date,
                    'timeline': [],
                    'message': 'データがありません'
                })
        
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
        # Config.pyから設定済みユーザー一覧を取得
        config = Config()
        users = config.get_all_users()
        
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

# ===== DEBUG API ENDPOINTS =====

@app.route('/api/debug/dice-scheduler/status', methods=['GET'])
def debug_dice_scheduler_status():
    """DiCEスケジューラーの状態とスケジュール確認API"""
    try:
        now = datetime.now()
        next_run = now.replace(hour=21, minute=0, second=0, microsecond=0)
        if now > next_run:
            next_run += timedelta(days=1)
        
        status = {
            'scheduler_running': dice_scheduler_running,
            'current_time': now.isoformat(),
            'next_scheduled_run': next_run.isoformat(),
            'seconds_until_next_run': (next_run - now).total_seconds(),
            'last_dice_result': last_dice_result,
            'scheduler_thread_alive': dice_scheduler_thread.is_alive() if dice_scheduler_thread else False
        }
        
        return jsonify({
            'status': 'success',
            'data': status
        })
    except Exception as e:
        logger.error(f"DiCEスケジューラー状態確認エラー: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/debug/dice-scheduler/trigger', methods=['POST'])
def debug_trigger_dice():
    """手動でDiCE改善提案を実行するデバッグAPI"""
    global last_dice_result
    
    try:
        data = request.get_json() or {}
        user_id = data.get('user_id', 'default')
        target_date = data.get('date')
        
        if target_date:
            if isinstance(target_date, str):
                target_date = datetime.fromisoformat(target_date.replace('Z', '+00:00')).date()
        else:
            target_date = datetime.now().date()
        
        logger.info(f"手動DiCE実行開始: ユーザー={user_id}, 日付={target_date}")
        
        dice_result = run_daily_dice_for_user(user_id)
        
        if dice_result:
            last_dice_result = {
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'result': dice_result,
                'execution_type': 'manual_debug'
            }
            
            return jsonify({
                'status': 'success',
                'message': '手動DiCE実行完了',
                'data': {
                    'user_id': user_id,
                    'execution_time': last_dice_result['timestamp'],
                    'total_improvement': dice_result.get('total_improvement', 0),
                    'schedule_items': len(dice_result.get('hourly_schedule', [])),
                    'dice_message': dice_result.get('message', ''),
                    'confidence': dice_result.get('confidence', 0),
                    'summary': dice_result.get('summary', ''),
                    'full_result': dice_result
                }
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'DiCE実行に失敗しました。データが不足している可能性があります。'
            }), 400
            
    except Exception as e:
        logger.error(f"手動DiCE実行エラー: {e}")
        return jsonify({
            'status': 'error',
            'message': f'エラーが発生しました: {str(e)}'
        }), 500

@app.route('/api/debug/dice-scheduler/results', methods=['GET'])
def debug_get_dice_results():
    """最新のDiCE結果を取得するデバッグAPI"""
    try:
        if not last_dice_result:
            return jsonify({
                'status': 'success',
                'message': 'DiCE結果がまだありません',
                'data': None
            })

        return jsonify({
            'status': 'success',
            'data': last_dice_result
        })
    except Exception as e:
        logger.error(f"DiCE結果取得エラー: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/debug/data-monitor/status', methods=['GET'])
def debug_data_monitor_status():
    """データ監視スレッドの状態確認API"""
    try:
        status = {
            'monitor_running': data_monitor_running,
            'monitor_thread_alive': data_monitor_thread.is_alive() if data_monitor_thread else False,
            'last_prediction': last_prediction_result if last_prediction_result else None,
            'current_time': datetime.now().isoformat()
        }

        return jsonify({
            'status': 'success',
            'data': status
        })
    except Exception as e:
        logger.error(f"データ監視状態確認エラー: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/debug/data-monitor/check', methods=['POST'])
def debug_trigger_data_check():
    """手動でデータ更新チェックを実行するデバッグAPI"""
    try:
        data = request.get_json() or {}
        user_id = data.get('user_id', 'default')

        has_new = sheets_connector.has_new_data(user_id)

        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'has_new_data': has_new,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"データ更新チェックエラー: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ===== HELPER FUNCTIONS =====

def get_user_config(user_id: str) -> Dict:
    """ユーザー設定を取得"""
    # main.pyのusers配列と同じ設定を取得
    users_config = [
        {
            'user_id': 'default', 
            'name': 'デフォルトユーザー', 
            'icon': '👤',
            'activity_sheet': 'Ua06e990fd6d5f4646615595d4e8d337f',
            'fitbit_sheet': 'kotoomi_Fitbit-data-kotomi'
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

def daily_dice_scheduler():
    """毎日21:00にDiCE改善提案を生成するスケジューラー"""
    global dice_scheduler_running, last_dice_result
    
    while dice_scheduler_running:
        try:
            now = datetime.now()
            # 毎日21:00に実行
            target_time = now.replace(hour=21, minute=0, second=0, microsecond=0)
            
            # 今日の21:00がまだ来ていない場合はそのまま、過ぎている場合は明日の21:00
            if now > target_time:
                target_time += timedelta(days=1)
            
            sleep_seconds = (target_time - now).total_seconds()
            logger.info(f"次回DiCE実行予定: {target_time.strftime('%Y-%m-%d %H:%M:%S')} ({sleep_seconds:.0f}秒後)")
            
            # 指定時刻まで待機
            time.sleep(sleep_seconds)
            
            if dice_scheduler_running:  # スケジューラーが停止されていないか確認
                logger.info("定時DiCE改善提案を実行中...")
                
                # デフォルトユーザーでDiCE実行
                user_id = 'default'
                dice_result = run_daily_dice_for_user(user_id)
                
                if dice_result:
                    last_dice_result = {
                        'timestamp': datetime.now().isoformat(),
                        'user_id': user_id,
                        'result': dice_result,
                        'execution_type': 'scheduled'
                    }
                    logger.info(f"定時DiCE実行完了: 改善ポイント {dice_result.get('total_improvement', 0):.1f}点")
                else:
                    logger.error("定時DiCE実行に失敗しました")
                    
        except Exception as e:
            logger.error(f"DiCEスケジューラーエラー: {e}")
            time.sleep(3600)  # エラー時は1時間待機

def run_daily_dice_for_user(user_id: str):
    """指定ユーザーの日次DiCE改善提案を実行"""
    try:
        # モデルが訓練されていることを確認
        activity_data = sheets_connector.get_activity_data(user_id)
        fitbit_data = sheets_connector.get_fitbit_data(user_id)

        if activity_data.empty:
            logger.warning(f"ユーザー {user_id} の活動データが見つかりません")
            return None

        # データ前処理とモデル訓練
        enhanced_data = predictor.preprocess_activity_data(activity_data)
        if not enhanced_data.empty:
            enhanced_data = predictor.aggregate_fitbit_by_activity(enhanced_data, fitbit_data)

            # 最新のモデルで訓練
            predictor.walk_forward_validation_train(enhanced_data)

        # 昨日のデータでDiCE実行
        yesterday = datetime.now() - timedelta(days=1)
        dice_result = explainer.generate_hourly_alternatives(enhanced_data, predictor, yesterday)

        return dice_result

    except Exception as e:
        logger.error(f"ユーザー {user_id} のDiCE実行エラー: {e}")
        return None

def data_monitor_loop():
    """
    データ更新を監視し、新しいデータが追加されたら自動的にフラストレーション予測を実行
    """
    global data_monitor_running, last_prediction_result

    check_interval = 600  # 600秒（10分）ごとにチェック

    while data_monitor_running:
        try:
            # デフォルトユーザーの新データをチェック
            user_id = 'default'

            if sheets_connector.has_new_data(user_id):
                logger.info(f"新しいデータを検知しました。フラストレーション予測を実行します: {user_id}")

                # 新しいデータを取得（キャッシュをクリアして最新データを取得）
                activity_data = sheets_connector.get_activity_data(user_id, use_cache=False)
                fitbit_data = sheets_connector.get_fitbit_data(user_id, use_cache=False)

                if not activity_data.empty:
                    # データ前処理
                    activity_processed = predictor.preprocess_activity_data(activity_data)
                    df_enhanced = predictor.aggregate_fitbit_by_activity(activity_processed, fitbit_data)

                    # モデル再訓練
                    if len(df_enhanced) > 10:
                        training_results = predictor.walk_forward_validation_train(df_enhanced)
                        logger.info(f"モデル再訓練完了: {training_results}")

                    # 最新の活動に対するフラストレーション予測
                    latest_activity = activity_processed.iloc[-1]
                    prediction_result = predictor.predict_single_activity(
                        latest_activity.get('CatSub', 'unknown'),
                        latest_activity.get('Duration', 60),
                        latest_activity.get('Timestamp', datetime.now())
                    )

                    # 予測結果を保存
                    last_prediction_result = {
                        'timestamp': datetime.now().isoformat(),
                        'user_id': user_id,
                        'latest_activity': latest_activity.get('CatSub', 'unknown'),
                        'prediction': prediction_result,
                        'data_count': len(df_enhanced)
                    }

                    logger.info(f"自動予測完了: {prediction_result}")

                    # 予測結果をスプレッドシートに保存
                    prediction_data = {
                        'timestamp': datetime.now().isoformat(),
                        'user_id': user_id,
                        'activity': latest_activity.get('CatSub', 'unknown'),
                        'duration': latest_activity.get('Duration', 0),
                        'predicted_frustration': prediction_result.get('predicted_frustration', 0),
                        'confidence': prediction_result.get('confidence', 0),
                        'actual_frustration': latest_activity.get('NASA_F', None)
                    }
                    sheets_connector.save_prediction_data(prediction_data)

            # 次のチェックまで待機
            time.sleep(check_interval)

        except Exception as e:
            logger.error(f"データ監視ループエラー: {e}")
            time.sleep(check_interval)

def initialize_application():
    """アプリケーション初期化"""
    global dice_scheduler_thread, dice_scheduler_running, data_monitor_thread, data_monitor_running

    try:
        logger.info("アプリケーションを初期化しています...")

        # スケジューラー開始
        scheduler.start_scheduler()
        logger.info("定期フィードバックスケジューラーを開始しました")

        # DiCE daily scheduler開始
        dice_scheduler_running = True
        dice_scheduler_thread = threading.Thread(target=daily_dice_scheduler, daemon=True)
        dice_scheduler_thread.start()
        logger.info("DiCE日次スケジューラーを開始しました (毎日21:00実行)")

        # データ更新監視スレッド開始
        data_monitor_running = True
        data_monitor_thread = threading.Thread(target=data_monitor_loop, daemon=True)
        data_monitor_thread.start()
        logger.info("データ更新監視スレッドを開始しました (10分ごとにチェック)")

        logger.info("アプリケーション初期化完了")
    except Exception as e:
        logger.error(f"アプリケーション初期化エラー: {e}")

def cleanup_application():
    """アプリケーション終了処理"""
    global dice_scheduler_running, data_monitor_running

    try:
        logger.info("アプリケーションを終了しています...")

        # DiCE スケジューラー停止
        dice_scheduler_running = False

        # データ監視スレッド停止
        data_monitor_running = False

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