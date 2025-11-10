"""
フラストレーション値予測・反実仮想説明システム
行動変更タイミングでの予測とDiCEによる改善提案を提供するFlaskアプリケーション
"""

from flask import Flask, render_template, jsonify, request
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict
import threading
import time
import json

# 日本標準時（JST）のタイムゾーン
JST = ZoneInfo('Asia/Tokyo')

from ml_model import FrustrationPredictor
from sheets_connector import SheetsConnector
from counterfactual_explainer import ActivityCounterfactualExplainer
from llm_feedback_generator import LLMFeedbackGenerator
from scheduler import FeedbackScheduler
from config import Config

app = Flask(__name__)

# Cloud Run用ログ設定
config = Config()

# 構造化ログ用のフォーマッター
class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            'severity': record.levelname,
            'message': record.getMessage(),
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        if hasattr(record, 'user_id'):
            log_obj['user_id'] = record.user_id
        if hasattr(record, 'endpoint'):
            log_obj['endpoint'] = record.endpoint
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
        return json.dumps(log_obj)

# ログレベル設定
# デバッグ用：一時的にINFOレベルに設定（DiCEとApp予測の調査のため）
log_level_str = os.environ.get('LOG_LEVEL', config.LOG_LEVEL)
if log_level_str == 'WARNING' and not config.IS_CLOUD_RUN:
    # ローカル環境では自動的にINFOレベルにしてデバッグログを表示
    log_level = logging.INFO
    print("⚠️ デバッグモード: ログレベルをINFOに設定しました")
else:
    log_level = getattr(logging, log_level_str.upper(), logging.WARNING)

# Cloud Run環境では構造化ログを使用
if config.IS_CLOUD_RUN:
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredFormatter())
    logging.basicConfig(level=log_level, handlers=[handler])
else:
    # ローカル環境では標準フォーマット
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

logger = logging.getLogger(__name__)

# Flask標準ログの抑制（HTTPアクセスログを無効化）
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
log.disabled = True  # 完全に無効化

# GUnicornアクセスログも抑制
gunicorn_logger = logging.getLogger('gunicorn.access')
gunicorn_logger.disabled = True
gunicorn_error_logger = logging.getLogger('gunicorn.error')
gunicorn_error_logger.setLevel(logging.ERROR)

# グローバルインスタンス
sheets_connector = SheetsConnector()
explainer = ActivityCounterfactualExplainer()
feedback_generator = LLMFeedbackGenerator()

# ユーザーごとのモデル管理
user_predictors = {}  # {user_id: FrustrationPredictor}

# schedulerにuser_predictorsを渡す（data_monitor_loopで学習済みのモデルを共有）
scheduler = FeedbackScheduler(user_predictors=user_predictors)
logger.warning(f"🔧 main.py: user_predictors辞書のID={id(user_predictors)}")

# アプリケーション初期化フラグ
_app_initialized = False

def get_predictor(user_id: str) -> FrustrationPredictor:
    """
    ユーザーごとのpredictorを取得（存在しない場合は作成）
    """
    if user_id not in user_predictors:
        logger.warning(f"🆕 新しいpredictorを作成: user_id={user_id}, 辞書ID={id(user_predictors)}")
        user_predictors[user_id] = FrustrationPredictor()
        logger.warning(f"✅ predictor作成完了: keys={list(user_predictors.keys())}")
    return user_predictors[user_id]

def ensure_model_trained(user_id: str, force_retrain: bool = False) -> dict:
    """
    ユーザーのモデルが訓練されていることを確認
    必要に応じて自動訓練を実行

    Args:
        user_id: ユーザーID
        force_retrain: 強制再訓練

    Returns:
        訓練結果または状態情報
    """
    predictor = get_predictor(user_id)

    # モデルが訓練済みで強制再訓練でない場合はスキップ
    logger.warning(f"🔍 ensure_model_trained: user_id={user_id}, predictor.model is None={predictor.model is None}, force_retrain={force_retrain}")
    if predictor.model is not None and not force_retrain:
        logger.warning(f"✅ モデルは既に訓練済みです: user_id={user_id}")
        return {
            'status': 'already_trained',
            'message': 'モデルは既に訓練済みです'
        }

    logger.warning(f"🎓 モデル訓練を開始します: user_id={user_id}")

    # データ取得
    activity_data = sheets_connector.get_activity_data(user_id)
    fitbit_data = sheets_connector.get_fitbit_data(user_id)

    if activity_data.empty:
        return {
            'status': 'no_data',
            'message': 'データがありません',
            'user_id': user_id
        }

    # データ前処理
    activity_processed = predictor.preprocess_activity_data(activity_data)
    df_enhanced = predictor.aggregate_fitbit_by_activity(activity_processed, fitbit_data)

    # データ品質チェック
    data_quality = predictor.check_data_quality(df_enhanced)

    if len(df_enhanced) < 10:
        return {
            'status': 'insufficient_data',
            'message': f'データ不足: {len(df_enhanced)}件 < 10件',
            'user_id': user_id,
            'data_count': len(df_enhanced),
            'data_quality': data_quality
        }

    # モデル訓練
    try:
        training_results = predictor.walk_forward_validation_train(df_enhanced)

        # 訓練結果にエラーが含まれていないかチェック
        if 'error' in training_results:
            logger.error(f"モデル訓練失敗: user_id={user_id}, error={training_results['error']}")
            return {
                'status': 'error',
                'message': f"訓練失敗: {training_results['error']}",
                'user_id': user_id,
                'data_quality': data_quality
            }

        # モデルが正常に初期化されたか最終確認
        if predictor.model is None:
            logger.error(f"❌ モデル訓練後もmodelがNoneです: user_id={user_id}")
            logger.error(f"   training_results: {training_results}")
            return {
                'status': 'error',
                'message': 'モデルの初期化に失敗しました',
                'user_id': user_id,
                'data_quality': data_quality
            }

        logger.warning(f"✅✅✅ モデル訓練完了: user_id={user_id}, "
                   f"RMSE={training_results.get('walk_forward_rmse', 0):.4f}, "
                   f"R²={training_results.get('walk_forward_r2', 0):.3f}, "
                   f"model_type={type(predictor.model).__name__}")
        return {
            'status': 'success',
            'message': 'モデル訓練完了',
            'user_id': user_id,
            'data_count': len(df_enhanced),
            'training_results': training_results,
            'data_quality': data_quality
        }
    except Exception as e:
        logger.error(f"モデル訓練エラー: user_id={user_id}, error={e}", exc_info=True)
        return {
            'status': 'error',
            'message': f'訓練エラー: {str(e)}',
            'user_id': user_id
        }

# データ更新監視スレッド
data_monitor_thread = None
data_monitor_running = False
last_prediction_result = {}  # 全ユーザーの予測結果を保存: {user_id: prediction_data}

def check_fitbit_data_availability(row):
    """
    Fitbitデータが利用可能かチェックする
    予測モデルが使用するSDNN_scaledとLorenz_Area_scaledが存在するかを確認
    """
    try:
        # 予測モデルが使用する必須カラム
        sdnn_scaled = row.get('SDNN_scaled')
        lorenz_area_scaled = row.get('Lorenz_Area_scaled')

        # 両方が有効な数値であることを確認
        sdnn_valid = False
        lorenz_valid = False

        if sdnn_scaled is not None and pd.notna(sdnn_scaled):
            try:
                float_val = float(sdnn_scaled)
                # NaN, Inf, -Infでない有効な数値か確認
                if not (np.isnan(float_val) or np.isinf(float_val)):
                    sdnn_valid = True
            except (ValueError, TypeError):
                pass

        if lorenz_area_scaled is not None and pd.notna(lorenz_area_scaled):
            try:
                float_val = float(lorenz_area_scaled)
                # NaN, Inf, -Infでない有効な数値か確認
                if not (np.isnan(float_val) or np.isinf(float_val)):
                    lorenz_valid = True
            except (ValueError, TypeError):
                pass

        is_available = sdnn_valid and lorenz_valid

        if not is_available:
            logger.debug(f"生体データ不足: SDNN_scaled={sdnn_valid}, Lorenz_Area_scaled={lorenz_valid}")

        return is_available

    except Exception as e:
        logger.warning(f"Fitbitデータ可用性チェックエラー: {e}")
        return False

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

@app.route('/mobile')
def mobile_mirror():
    """スマホ専用UI - 手動ユーザー選択・縦スクロール表示"""
    return render_template('mobile_mirror.html')

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
        
        # ユーザーごとのpredictorを取得
        predictor = get_predictor(user_id)

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

        # データ品質チェック
        data_quality = predictor.check_data_quality(df_enhanced)

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

            # 訓練が失敗した場合はエラーを返す
            if training_result.get('status') != 'success':
                return jsonify({
                    'status': 'error',
                    'message': f"モデルの訓練に失敗しました: {training_result.get('message')}",
                    'user_id': user_id,
                    'data_quality': data_quality,
                    'training_result': training_result
                }), 400

        # フラストレーション値予測
        prediction_result = predictor.predict_frustration_at_activity_change(
            df_enhanced, target_timestamp
        )

        # モデルパフォーマンス情報
        training_results = training_info if training_info['auto_trained'] else {'status': 'already_trained'}

        # 行動変更タイミング一覧
        change_timestamps = predictor.get_activity_change_timestamps(df_enhanced)

        response = {
            'status': 'success',
            'user_id': user_id,
            'prediction': prediction_result,
            'activity_change_timestamps': [ts.isoformat() for ts in change_timestamps],
            'model_performance': training_results,
            'data_quality': data_quality,
            'timestamp': datetime.now().isoformat()
        }

        # データ不足時の警告メッセージを追加
        if not data_quality['is_sufficient']:
            response['warning'] = {
                'message': 'データが不足しているため、予測精度が低い可能性があります。',
                'details': data_quality['warnings'],
                'recommendations': data_quality['recommendations']
            }

        return jsonify(response)
        
    except Exception as e:
        logger.error(f"フラストレーション値予測エラー: {e}")
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

        # ユーザーごとのpredictorを取得
        predictor = get_predictor(user_id)

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

        # モデルが訓練されていることを確認（DiCE実行前に必須）
        training_result = ensure_model_trained(user_id)
        if training_result.get('status') not in ['success', 'already_trained']:
            return jsonify({
                'status': 'error',
                'message': f"モデルの訓練に失敗しました: {training_result.get('message')}",
                'user_id': user_id,
                'training_result': training_result
            }), 400

        # 1時間ごとのスケジュール提案を生成（DiCE方式）
        dice_result = explainer.generate_hourly_alternatives(
            df_enhanced,  # 前処理済みデータを渡す
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
    Hourly LogからDiCE提案を取得するAPI（DiCE実行はしない）
    スケジューラーが23:55 JST（14:55 UTC）に実行したDiCE結果を読み取るだけ
    """
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default')
        date = data.get('date', datetime.now().strftime('%Y-%m-%d'))

        # ログレベルをDEBUGに変更（頻繁に呼ばれるため、WARNINGレベルでは多すぎる）
        if config.ENABLE_DEBUG_LOGS:
            logger.debug(f"📊 DiCE分析取得: user_id={user_id}, date={date}")

        # Hourly LogからDiCE提案を取得
        hourly_log = sheets_connector.get_hourly_log(user_id, date)

        if hourly_log.empty:
            return jsonify({
                'status': 'success',
                'user_id': user_id,
                'dice_analysis': {
                    'type': 'no_data',
                    'timeline': [],
                    'summary': 'まだDiCE提案が生成されていません。23:55以降に確認してください。'
                },
                'timestamp': datetime.now().isoformat()
            })

        # Activity_Dataを取得（Durationを取得するため）
        activity_data = sheets_connector.get_activity_data(user_id)

        # DiCE提案があるデータのみ抽出
        dice_suggestions = []
        for idx, row in hourly_log.iterrows():
            dice_suggestion = row.get('DiCE提案活動名')
            if pd.notna(dice_suggestion) and dice_suggestion != '':
                time_str = row.get('時刻')
                activity = row.get('活動名')
                predicted_f = row.get('予測NASA_F')
                improvement = row.get('改善幅')
                improved_f = row.get('改善後F値')

                # タイムスタンプを作成（フロントエンドがDate型として扱えるように）
                timestamp_str = f"{date} {time_str}:00" if time_str else f"{date} 00:00:00"

                # 改善幅を数値化
                improvement_value = float(improvement) if pd.notna(improvement) else 0

                # time_rangeを計算（Activity_DataからDurationを取得）
                time_range = time_str  # デフォルト
                duration_minutes = 60  # デフォルト
                try:
                    if not activity_data.empty and 'Timestamp' in activity_data.columns:
                        # 該当する活動を探す
                        matching = activity_data[
                            (activity_data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M') == f"{date} {time_str}") &
                            (activity_data['CatSub'] == activity)
                        ]
                        if not matching.empty:
                            duration_minutes = int(matching.iloc[0].get('Duration', 60))
                            # time_rangeを計算
                            from datetime import datetime as dt_class, timedelta
                            start_time = dt_class.strptime(f"{date} {time_str}", '%Y-%m-%d %H:%M')
                            end_time = start_time + timedelta(minutes=duration_minutes)
                            time_range = f"{time_str}-{end_time.strftime('%H:%M')}"
                except Exception as e:
                    logger.debug(f"time_range計算エラー（デフォルト値使用）: {e}")

                dice_suggestions.append({
                    # フロントエンド用フィールド（tablet_mirror.html）
                    'timestamp': timestamp_str,
                    'frustration_reduction': improvement_value,
                    # バックエンド互換性用フィールド（既存コード用）
                    'time': time_str,
                    'improvement': improvement_value,
                    # 共通フィールド
                    'original_activity': activity,
                    'original_frustration': float(predicted_f) if pd.notna(predicted_f) else None,
                    'suggested_activity': dice_suggestion,
                    'improved_frustration': float(improved_f) if pd.notna(improved_f) else None,
                    'time_range': time_range  # 追加
                })

        # DiCE分析結果を構築
        if len(dice_suggestions) > 0:
            dice_result = {
                'type': 'dice_analysis',
                'timeline': dice_suggestions,
                'summary': f'{len(dice_suggestions)}件の改善提案があります。',
                'total_improvement': sum([s.get('improvement', 0) or 0 for s in dice_suggestions])
            }
        else:
            dice_result = {
                'type': 'no_suggestions',
                'timeline': [],
                'summary': 'DiCE提案はまだ生成されていません。23:55以降に確認してください。'
            }

        # ログレベルをDEBUGに変更（頻繁に呼ばれるため）
        if config.ENABLE_DEBUG_LOGS:
            logger.debug(f"✅ DiCE提案取得完了: {len(dice_suggestions)}件")

        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'dice_analysis': dice_result,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"DiCE分析取得エラー: {e}")
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
        
        if config.ENABLE_DEBUG_LOGS:
            logger.debug(f"Timeline API呼び出し - user_id: {user_id}, date: {date}")
        
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
                    'is_new_user': True,
                    'message': 'Google Sheetsに接続できません。認証設定を確認してください。'
                })
            else:
                return jsonify({
                    'status': 'success',
                    'date': date,
                    'timeline': [],
                    'is_new_user': True,
                    'message': 'データがありません',
                    'warning': {
                        'message': '新規ユーザーまたはデータ未記録です。',
                        'recommendations': [
                            '活動データを記録してください。',
                            'データが蓄積されると、フラストレーション予測とDiCE提案が利用可能になります。'
                        ]
                    }
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
        
        # Hourly Logから予測済みデータを取得（F値、DiCE提案を含む）
        hourly_log = sheets_connector.get_hourly_log(user_id, date)

        # タイムライン作成（活動データ + Hourly Logマージ）
        timeline = []
        for idx, row in daily_data.iterrows():
            timestamp = row['Timestamp']
            time_str = timestamp.strftime('%H:%M') if hasattr(timestamp, 'strftime') else str(timestamp)
            activity_name = row.get('CatSub', '')

            # 活動名のバリデーション
            if not activity_name or pd.isna(activity_name):
                logger.warning(f"活動名が不正です (CatSub='{activity_name}') @{time_str} - スキップします")
                continue

            # Hourly Logから予測値・DiCE提案を取得
            # 【重要】Hourly Logに登録されている活動のみを表示
            if hourly_log.empty:
                continue  # Hourly Logが空なら全てスキップ

            cached = hourly_log[
                (hourly_log['時刻'] == time_str) &
                (hourly_log['活動名'] == activity_name)
            ]

            if cached.empty:
                # Hourly Logに未登録の活動はスキップ（デフォルト値10を表示しないため）
                continue

            # Hourly Logから予測値とDiCE提案を取得
            cached_row = cached.iloc[0]
            predicted_frustration = cached_row.get('予測NASA_F')
            dice_suggestion = cached_row.get('DiCE提案活動名')
            improvement = cached_row.get('改善幅')
            improved_frustration = cached_row.get('改善後F値')

            # NaNチェック
            if pd.isna(predicted_frustration):
                predicted_frustration = None
            if pd.isna(dice_suggestion) or dice_suggestion == '':
                dice_suggestion = None
            if pd.isna(improvement):
                improvement = None
            if pd.isna(improved_frustration):
                improved_frustration = None

            # F値変換
            frustration_for_timeline = float(predicted_frustration) if predicted_frustration is not None else None

            # タイムラインに追加（Hourly Logに登録済みの活動のみ）
            duration_minutes = int(row.get('Duration', 60))
            timeline_entry = {
                'timestamp': timestamp.isoformat(),
                'hour': timestamp.hour if hasattr(timestamp, 'hour') else 0,
                'activity': activity_name,
                'duration': duration_minutes,
                'frustration_value': frustration_for_timeline,
                'is_predicted': predicted_frustration is not None
            }

            # DiCE提案がある場合は追加
            if dice_suggestion:
                # time_rangeを計算（例: "13:00-16:00"）
                from datetime import timedelta
                end_time = timestamp + timedelta(minutes=duration_minutes)
                time_range = f"{time_str}-{end_time.strftime('%H:%M')}"

                timeline_entry['dice_suggestion'] = {
                    'suggested_activity': dice_suggestion,
                    'improvement': float(improvement) if improvement is not None else None,
                    'improved_frustration': float(improved_frustration) if improved_frustration is not None else None,
                    'time_range': time_range  # 追加
                }

            timeline.append(timeline_entry)
        
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
    Daily SummaryからChatGPTフィードバックを読み取るAPI（生成しない）

    フィードバック生成は1日1回、scheduler.py の _execute_evening_feedback で実行される（22:50 JST）
    このAPIはDaily Summaryシートから既に生成されたフィードバックを読み取るだけ
    """
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default')
        feedback_type = data.get('feedback_type', data.get('type', 'daily'))

        # 今日の日付を取得
        today = datetime.now().strftime('%Y-%m-%d')

        logger.info(f"📖 Daily Summaryからフィードバック取得: user_id={user_id}, date={today}")

        # Daily Summaryシートからフィードバックを取得
        summary = sheets_connector.get_daily_summary(user_id, today)

        if not summary:
            return jsonify({
                'status': 'success',
                'user_id': user_id,
                'feedback_type': 'daily',
                'feedback': {
                    'main_feedback': 'フィードバックはまだ生成されていません。22:50以降に確認してください。',
                    'action_plan': [],
                    'total_improvement_potential': 0,
                    'num_suggestions': 0,
                    'generated_at': None
                },
                'message': 'フィードバックは毎日22:50（JST）に生成されます。'
            })

        # Daily Summaryのデータをフィードバック形式に変換
        feedback_result = {
            'main_feedback': summary.get('chatgpt_feedback', ''),
            'action_plan': summary.get('action_plan', []),
            'total_improvement_potential': summary.get('dice_improvement', 0),
            'num_suggestions': summary.get('dice_count', 0),
            'generated_at': summary.get('generated_at', '')
        }

        logger.info(f"✅ Daily Summaryからフィードバック取得完了: user_id={user_id}")

        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'feedback_type': 'daily',
            'feedback': feedback_result,
            'daily_stats': {
                'avg_predicted': round(summary.get('avg_predicted', 0), 2) if summary.get('avg_predicted') is not None else None,
                'dice_suggestions': summary.get('dice_count', 0)
            },
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"フィードバック取得エラー: {e}")
        import traceback
        logger.error(traceback.format_exc())
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

# ==========================================
# Cloud Scheduler用エンドポイント（min_instances=0用）
# ==========================================

@app.route('/api/scheduler/monitor', methods=['POST'])
def trigger_cloud_scheduler_monitor():
    """
    Cloud Schedulerからデータ監視を起動（1回実行）
    既存のdata_monitor_loop()を使用する場合は、このエンドポイントは使用しない
    """
    try:
        # 簡易認証: ヘッダーチェック（オプション）
        # Cloud Schedulerでは OIDC トークンを使用することを推奨
        auth_header = request.headers.get('X-Scheduler-Auth', '')
        expected_auth = os.environ.get('SCHEDULER_AUTH_TOKEN', 'default-scheduler-token')

        if auth_header != expected_auth:
            logger.warning(f"⚠️ Cloud Scheduler認証失敗: ヘッダー={auth_header[:10]}...")
            return jsonify({
                'status': 'error',
                'message': '認証に失敗しました'
            }), 401

        logger.warning(f"🔍 Cloud Schedulerからデータ監視リクエストを受信: {datetime.now(JST).strftime('%Y-%m-%d %H:%M')}")

        # データ監視を1回実行
        result = run_data_monitor_once()

        return jsonify({
            'status': 'success',
            'message': 'データ監視を実行しました',
            'result': result,
            'timestamp': datetime.now(JST).isoformat()
        })

    except Exception as e:
        logger.error(f"Cloud Schedulerデータ監視エラー: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/scheduler/dice', methods=['POST'])
def trigger_cloud_scheduler_dice():
    """
    Cloud SchedulerからDiCE実行を起動（1回実行）
    既存のscheduler._execute_evening_feedback()と同等の処理
    """
    try:
        # 簡易認証: ヘッダーチェック
        auth_header = request.headers.get('X-Scheduler-Auth', '')
        expected_auth = os.environ.get('SCHEDULER_AUTH_TOKEN', 'default-scheduler-token')

        if auth_header != expected_auth:
            logger.warning(f"⚠️ Cloud Scheduler認証失敗: ヘッダー={auth_header[:10]}...")
            return jsonify({
                'status': 'error',
                'message': '認証に失敗しました'
            }), 401

        logger.warning(f"🎲 Cloud SchedulerからDiCE実行リクエストを受信: {datetime.now(JST).strftime('%Y-%m-%d %H:%M')}")

        # scheduler._execute_evening_feedback()を直接呼び出し
        scheduler._execute_evening_feedback()

        return jsonify({
            'status': 'success',
            'message': 'DiCE実行を完了しました',
            'timestamp': datetime.now(JST).isoformat()
        })

    except Exception as e:
        logger.error(f"Cloud Scheduler DiCE実行エラー: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ==========================================
# 既存エンドポイント
# ==========================================

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
        user_id = request.args.get('user_id', 'default')

        # ユーザーごとのpredictorを取得
        predictor = get_predictor(user_id)

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

@app.route('/api/debug/model', methods=['GET'])
def debug_model():
    """モデルデバッグ情報取得API"""
    try:
        user_id = request.args.get('user_id', 'default')

        # ユーザーごとのpredictorを取得
        predictor = get_predictor(user_id)

        # データ取得
        activity_data = sheets_connector.get_activity_data(user_id)
        fitbit_data = sheets_connector.get_fitbit_data(user_id)

        # データ前処理
        activity_processed = predictor.preprocess_activity_data(activity_data)
        df_enhanced = predictor.aggregate_fitbit_by_activity(activity_processed, fitbit_data)

        # データ品質
        data_quality = predictor.check_data_quality(df_enhanced)

        # NASA_F統計
        nasa_f_stats = None
        if not df_enhanced.empty and 'NASA_F' in df_enhanced.columns:
            nasa_f = df_enhanced['NASA_F'].dropna()
            if not nasa_f.empty:
                nasa_f_stats = {
                    'count': int(len(nasa_f)),
                    'mean': float(nasa_f.mean()),
                    'std': float(nasa_f.std()),
                    'min': float(nasa_f.min()),
                    'max': float(nasa_f.max()),
                    'unique_values': int(nasa_f.nunique()),
                    'value_distribution': nasa_f.value_counts().head(10).to_dict()
                }

        # 活動統計
        activity_stats = None
        if not df_enhanced.empty and 'CatSub' in df_enhanced.columns:
            activities = df_enhanced['CatSub'].dropna()
            if not activities.empty:
                activity_stats = {
                    'total': int(len(activities)),
                    'unique': int(activities.nunique()),
                    'top_5': activities.value_counts().head(5).to_dict()
                }

        # モデル状態
        model_info = {
            'is_trained': predictor.model is not None,
            'feature_count': len(predictor.feature_columns) if predictor.feature_columns else 0,
            'feature_names': predictor.feature_columns if predictor.feature_columns else []
        }

        # エンコーダー情報
        encoders_info = {}
        if hasattr(predictor, 'encoders'):
            for key, encoder in predictor.encoders.items():
                if hasattr(encoder, 'classes_'):
                    encoders_info[key] = {
                        'n_classes': len(encoder.classes_),
                        'classes': list(encoder.classes_)
                    }

        # Walk Forward Validation結果
        wfv_results = None
        if predictor.model is not None and len(df_enhanced) >= 10:
            try:
                training_results = predictor.walk_forward_validation_train(df_enhanced)
                wfv_results = {
                    'rmse': float(training_results.get('walk_forward_rmse', 0)),
                    'mae': float(training_results.get('walk_forward_mae', 0)),
                    'r2': float(training_results.get('walk_forward_r2', 0)),
                    'prediction_diversity': training_results.get('prediction_diversity', {}),
                    'feature_importance': {
                        k: float(v) for k, v in sorted(
                            training_results.get('feature_importance', {}).items(),
                            key=lambda x: x[1],
                            reverse=True
                        )[:15]  # 上位15特徴量
                    }
                }
            except Exception as e:
                logger.error(f"WFV実行エラー: {e}")
                wfv_results = {'error': str(e)}

        # テスト予測
        test_predictions = []
        if not df_enhanced.empty and predictor.model is not None:
            test_cases = df_enhanced.head(3)
            for idx, row in test_cases.iterrows():
                activity = row.get('CatSub')
                if activity is None:
                    activity = 'unknown'

                duration = row.get('Duration', 60)
                timestamp = pd.to_datetime(row.get('Timestamp'))
                actual = row.get('NASA_F')

                # 固定値予測
                pred1 = predictor.predict_single_activity(activity, duration, timestamp)

                # 履歴使用予測
                pred2 = predictor.predict_with_history(activity, duration, timestamp, df_enhanced)

                test_predictions.append({
                    'activity': activity,
                    'timestamp': timestamp.isoformat(),
                    'actual': float(actual) if pd.notna(actual) else None,
                    'predicted_fixed': float(pred1.get('predicted_frustration', 0)),
                    'predicted_history': float(pred2.get('predicted_frustration', 0)),
                    'historical_records': int(pred2.get('historical_records', 0))
                })

        # 診断
        issues = []
        if data_quality['total_samples'] < 10:
            issues.append(f"データ数不足: {data_quality['total_samples']}件")
        if nasa_f_stats and nasa_f_stats['std'] < 1.0:
            issues.append(f"NASA_Fの分散が小さい: {nasa_f_stats['std']:.2f}")
        if nasa_f_stats and nasa_f_stats['unique_values'] < 3:
            issues.append(f"NASA_Fの種類が少ない: {nasa_f_stats['unique_values']}種類")
        if not predictor.model:
            issues.append("モデル未訓練")
        if activity_stats and activity_stats['unique'] < 3:
            issues.append(f"活動種類が少ない: {activity_stats['unique']}種類")

        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'data_quality': data_quality,
            'nasa_f_stats': nasa_f_stats,
            'activity_stats': activity_stats,
            'model_info': model_info,
            'encoders_info': encoders_info,
            'wfv_results': wfv_results,
            'test_predictions': test_predictions,
            'issues': issues,
            'diagnosis': 'OK' if not issues else 'Issues detected'
        })

    except Exception as e:
        logger.error(f"モデルデバッグエラー: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/data/stats', methods=['GET'])
def get_data_stats():
    """データ統計情報取得API"""
    try:
        user_id = request.args.get('user_id', 'default')

        # ユーザーごとのpredictorを取得
        predictor = get_predictor(user_id)

        # データ取得
        activity_data = sheets_connector.get_activity_data(user_id)
        fitbit_data = sheets_connector.get_fitbit_data(user_id)

        # データ前処理
        activity_processed = predictor.preprocess_activity_data(activity_data)
        df_enhanced = predictor.aggregate_fitbit_by_activity(activity_processed, fitbit_data)

        # データ品質チェック
        data_quality = predictor.check_data_quality(df_enhanced)

        # NASA_F統計
        nasa_f_stats = {}
        if not activity_data.empty and 'NASA_F' in activity_data.columns:
            nasa_f_values = pd.to_numeric(activity_data['NASA_F'], errors='coerce').dropna()
            if not nasa_f_values.empty:
                nasa_f_stats = {
                    'count': int(len(nasa_f_values)),
                    'mean': float(nasa_f_values.mean()),
                    'std': float(nasa_f_values.std()),
                    'min': float(nasa_f_values.min()),
                    'max': float(nasa_f_values.max()),
                    'median': float(nasa_f_values.median()),
                    'unique_values': int(nasa_f_values.nunique())
                }

        # 活動カテゴリ統計
        activity_stats = {}
        if not activity_data.empty and 'CatSub' in activity_data.columns:
            activity_counts = activity_data['CatSub'].value_counts()
            activity_stats = {
                'total_activities': int(len(activity_data)),
                'unique_activities': int(activity_data['CatSub'].nunique()),
                'top_activities': activity_counts.head(10).to_dict()
            }

        # Fitbitデータ統計
        fitbit_stats = {
            'total_records': int(len(fitbit_data)) if not fitbit_data.empty else 0,
            'has_data': not fitbit_data.empty
        }

        # モデル統計
        model_stats = {
            'is_trained': predictor.model is not None,
            'feature_count': len(predictor.feature_columns) if hasattr(predictor, 'feature_columns') and predictor.feature_columns else 0,
            'training_history_count': len(predictor.walk_forward_history) if hasattr(predictor, 'walk_forward_history') and predictor.walk_forward_history else 0
        }

        # 日付範囲
        date_range = {}
        if not activity_data.empty and 'Timestamp' in activity_data.columns:
            timestamps = pd.to_datetime(activity_data['Timestamp'], errors='coerce').dropna()
            if not timestamps.empty:
                date_range = {
                    'earliest': timestamps.min().isoformat(),
                    'latest': timestamps.max().isoformat(),
                    'days_span': (timestamps.max() - timestamps.min()).days
                }

        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'data_quality': data_quality,
            'nasa_f_stats': nasa_f_stats,
            'activity_stats': activity_stats,
            'fitbit_stats': fitbit_stats,
            'model_stats': model_stats,
            'date_range': date_range,
            'summary': {
                'total_activity_records': int(len(activity_data)) if not activity_data.empty else 0,
                'processed_records': int(len(df_enhanced)) if not df_enhanced.empty else 0,
                'data_sufficient': data_quality.get('is_sufficient', False),
                'quality_level': data_quality.get('quality_level', 'unknown')
            }
        })

    except Exception as e:
        logger.error(f"データ統計取得エラー: {e}")
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

def calculate_and_save_daily_summary(user_id: str, target_date=None):
    """
    指定ユーザーの日次サマリーを計算してスプレッドシートに保存

    Args:
        user_id: ユーザーID
        target_date: 対象日付（デフォルトは昨日）

    Returns:
        保存成功: True, 失敗: False
    """
    try:
        if target_date is None:
            # デフォルトは昨日
            target_date = (datetime.now() - timedelta(days=1)).date()
        elif isinstance(target_date, str):
            target_date = datetime.fromisoformat(target_date).date()

        # データ取得
        activity_data = sheets_connector.get_activity_data(user_id, use_cache=False)

        if activity_data.empty:
            logger.warning(f"日次サマリー計算: ユーザー {user_id} のデータがありません")
            return False

        # 対象日のデータをフィルタリング
        activity_data['Date'] = pd.to_datetime(activity_data['Timestamp']).dt.date
        daily_data = activity_data[activity_data['Date'] == target_date]

        if daily_data.empty:
            logger.info(f"日次サマリー計算: {target_date} のデータがありません (ユーザー: {user_id})")
            return False

        # フラストレーション値の統計を計算
        frustration_values = daily_data['NASA_F'].dropna()

        if frustration_values.empty:
            logger.warning(f"日次サマリー計算: NASA_F値がありません (ユーザー: {user_id}, 日付: {target_date})")
            return False

        # サマリーデータを作成
        summary_data = {
            'avg_frustration': float(frustration_values.mean()),
            'min_frustration': float(frustration_values.min()),
            'max_frustration': float(frustration_values.max()),
            'activity_count': int(len(daily_data)),
            'total_duration': int(daily_data['Duration'].sum()),
            'unique_activities': int(daily_data['CatSub'].nunique()),
            'notes': f'Auto-generated from {len(daily_data)} activities'
        }

        # スプレッドシートに保存
        success = sheets_connector.save_daily_summary(
            user_id=user_id,
            date=target_date.isoformat(),
            summary_data=summary_data
        )

        if success and config.ENABLE_INFO_LOGS:
            logger.info(f"日次サマリー保存完了: {user_id}, {target_date}, "
                       f"平均: {summary_data['avg_frustration']:.2f}, "
                       f"活動数: {summary_data['activity_count']}")

        return success

    except Exception as e:
        logger.error(f"日次サマリー計算エラー: {e}")
        return False

# ===== DEBUG API ENDPOINTS =====

@app.route('/api/frustration/daily-summary', methods=['POST'])
def generate_daily_summary():
    """
    日次サマリー計算・保存API
    指定日（デフォルトは昨日）のフラストレーション統計を計算してスプレッドシートに保存
    """
    try:
        data = request.get_json() or {}
        user_id = data.get('user_id', 'default')
        target_date = data.get('date')

        # 日次サマリーを計算・保存
        success = calculate_and_save_daily_summary(user_id, target_date)

        if success:
            return jsonify({
                'status': 'success',
                'message': '日次サマリーを保存しました',
                'user_id': user_id,
                'date': target_date if target_date else (datetime.now() - timedelta(days=1)).date().isoformat(),
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': '日次サマリーの保存に失敗しました。データが不足している可能性があります。',
                'user_id': user_id
            }), 400

    except Exception as e:
        logger.error(f"日次サマリー生成エラー: {e}")
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

@app.route('/api/tablet/data/<user_id>', methods=['GET'])
def get_tablet_data(user_id):
    """タブレット用統合データAPI - すべてのデータを一度に取得"""
    try:
        if config.ENABLE_DEBUG_LOGS:
            logger.debug(f"タブレットデータAPI呼び出し - user_id: {user_id}")
        
        # 今日の日付を取得
        today = datetime.now().strftime('%Y-%m-%d')
        
        # 1. タイムラインデータを取得
        timeline_data = []
        try:
            with app.test_request_context(json={'user_id': user_id, 'date': today}):
                timeline_response = get_frustration_timeline()
                if hasattr(timeline_response, 'get_json'):
                    timeline_result = timeline_response.get_json()
                    if timeline_result and timeline_result.get('status') == 'success':
                        timeline_data = timeline_result.get('timeline', [])
        except Exception as timeline_error:
            logger.warning(f"タイムラインデータ取得警告: {timeline_error}")
        
        # 2. DiCE分析データとフィードバックデータをDaily Summaryから取得（キャッシュ）
        # 毎回DiCE実行するのではなく、既に保存されたDaily Summaryを使用
        dice_data = {}
        feedback_data = {}

        try:
            # Daily Summaryシートから今日のサマリーを取得
            sheet_name = f"{user_id}_Daily_Summary"
            worksheet = sheets_connector._find_worksheet_by_exact_name(sheet_name)

            if worksheet:
                all_values = worksheet.get_all_values()
                for row in all_values[1:]:  # ヘッダーをスキップ
                    if len(row) > 0 and row[0] == today:
                        # 今日のサマリーが見つかった
                        logger.info(f"Daily Summaryから取得: {today}")
                        feedback_data = {
                            'feedback': row[6] if len(row) > 6 else '',  # ChatGPTフィードバック
                            'action_plan': json.loads(row[7]) if len(row) > 7 and row[7] else []  # アクションプラン
                        }
                        dice_data = {
                            'improvement_potential': float(row[4]) if len(row) > 4 and row[4] else 0,  # DiCE改善ポテンシャル
                            'suggestion_count': int(row[5]) if len(row) > 5 and row[5] else 0  # DiCE提案数
                        }
                        logger.info(f"キャッシュから取得: DiCE={dice_data.get('suggestion_count')}件, Feedback={'あり' if feedback_data.get('feedback') else 'なし'}")
                        break

            # Daily Summaryにデータがない場合のみ、DiCE分析とフィードバック生成を実行
            if not dice_data and not feedback_data:
                logger.info(f"Daily Summaryが存在しないため、DiCE分析とフィードバック生成をスキップします: {today}")
                # スケジューラーが1日1回実行するため、ここでは空のデータを返す

        except Exception as cache_error:
            logger.warning(f"Daily Summaryキャッシュ取得エラー: {cache_error}")
        
        # 4. 今日の統計を計算
        daily_stats = {
            'date': today,
            'total_activities': len(timeline_data),
            'avg_frustration': 0,
            'min_frustration': 0,
            'max_frustration': 0
        }
        
        if timeline_data:
            frustration_values = [item.get('frustration_value', 0) for item in timeline_data if item.get('frustration_value') is not None]
            if frustration_values:
                daily_stats['avg_frustration'] = round(sum(frustration_values) / len(frustration_values), 1)
                daily_stats['min_frustration'] = min(frustration_values)
                daily_stats['max_frustration'] = max(frustration_values)
        
        # 5. 統合レスポンスを作成
        response_data = {
            'status': 'success',
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'daily_stats': daily_stats,
            'timeline': timeline_data,
            'dice_analysis': dice_data,
            'feedback': feedback_data,
            'system_info': {
                'data_source': 'spreadsheet',
                'last_update': datetime.now().isoformat(),
                'prediction_logging': True,
                'data_monitoring': True
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"タブレットデータAPI エラー: {e}")
        return jsonify({
            'status': 'error',
            'user_id': user_id,
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# ===== HELPER FUNCTIONS =====

def get_user_config(user_id: str) -> Dict:
    """
    ユーザー設定を取得
    Config.pyから設定を取得するように変更
    """
    config = Config()
    users = config.get_all_users()

    for user in users:
        if user['user_id'] == user_id:
            return user

    # 見つからない場合は最初のユーザーを返す
    if users:
        return users[0]

    return {'user_id': user_id, 'name': user_id}

def data_monitor_loop():
    """
    全ユーザーのデータを監視し、nasa_status='done'の活動を自動的に予測
    9:00-22:00の間、毎時0分と30分に実行
    """
    global data_monitor_running, last_prediction_result

    # 全ユーザーのリストをConfigから取得
    users_config = config.get_all_users()

    def get_next_run_time():
        """次の実行時刻を計算（JST: 9:00-22:00の間、毎時0分と30分）"""
        from datetime import timedelta

        now = datetime.now(JST)
        current_hour = now.hour
        current_minute = now.minute

        # 次の実行時刻候補を計算
        if current_minute < 30:
            # 今の時間の30分
            next_time = now.replace(minute=30, second=0, microsecond=0)
        else:
            # 次の時間の00分
            next_time = now + timedelta(hours=1)
            next_time = next_time.replace(minute=0, second=0, microsecond=0)

        # 9:00-22:00の範囲外の場合は翌日9:00に設定
        while next_time.hour < 9 or next_time.hour > 22:
            if next_time.hour > 22 or next_time.hour < 9:
                # 翌日9:00
                next_day = next_time + timedelta(days=1)
                next_time = next_day.replace(hour=9, minute=0, second=0, microsecond=0)
            else:
                break

        return next_time

    logger.warning(f"🕐 データ監視ループ開始: 9:00-22:00の間、毎時0分と30分に実行")

    while data_monitor_running:
        try:
            # 次の実行時刻まで待機
            next_run = get_next_run_time()
            wait_seconds = (next_run - datetime.now(JST)).total_seconds()

            if wait_seconds > 0:
                logger.warning(f"⏰ 次の実行時刻: {next_run.strftime('%H:%M')}, 待機時間: {int(wait_seconds)}秒")
                time.sleep(wait_seconds)

            logger.warning(f"🔍 データ監視開始: {datetime.now(JST).strftime('%Y-%m-%d %H:%M')}")

            # 全ユーザーをチェック
            for user_config in users_config:
                user_id = user_config['user_id']
                user_name = user_config['name']

                # 毎回全活動をチェック（has_new_data不要）
                try:
                    # ユーザーごとのpredictorを取得
                    predictor = get_predictor(user_id)

                    # データ取得（キャッシュなし）
                    activity_data = sheets_connector.get_activity_data(user_id, use_cache=False)
                    fitbit_data = sheets_connector.get_fitbit_data(user_id, use_cache=False)

                    if activity_data.empty:
                        logger.warning(f"活動データなし: {user_name}")
                        continue

                    # nasa_status='done'の行のみフィルタリング
                    if 'nasa_status' in activity_data.columns:
                        activity_data_done = activity_data[activity_data['nasa_status'] == 'done'].copy()
                        logger.warning(f"📊 {user_name}: 全活動={len(activity_data)}件, nasa_status='done'={len(activity_data_done)}件")
                    else:
                        # nasa_status列がない場合は全てを処理
                        activity_data_done = activity_data.copy()
                        logger.warning(f"⚠️ {user_name}: nasa_status列が見つかりません。全活動を処理します")

                    if activity_data_done.empty:
                        logger.warning(f"nasa_status='done'の活動なし: {user_name}")
                        continue

                    # モデル訓練確認
                    training_result = ensure_model_trained(user_id, force_retrain=False)
                    if training_result['status'] not in ['success', 'already_trained']:
                        logger.warning(f"モデル訓練失敗 ({user_name}): {training_result.get('message')}")
                        continue

                    if predictor.model is None:
                        logger.error(f"モデルが初期化されていません ({user_name})")
                        continue

                    # データ前処理
                    activity_processed = predictor.preprocess_activity_data(activity_data_done)
                    df_enhanced = predictor.aggregate_fitbit_by_activity(activity_processed, fitbit_data)

                    logger.warning(f"🔍 予測チェック開始: {user_name}, 対象活動={len(df_enhanced)}件")

                    # 【重要】全期間のHourly Logを一度に取得（効率化と重複防止）
                    all_dates = df_enhanced['Timestamp'].dt.strftime('%Y-%m-%d').unique()
                    hourly_log_cache = {}
                    for date in all_dates:
                        hourly_log_cache[date] = sheets_connector.get_hourly_log(user_id, date)

                    logger.warning(f"📋 Hourly Log取得完了: {len(hourly_log_cache)}日分")

                    # 新規活動のみを抽出
                    new_activities = []
                    update_predictions = []

                    for idx, row in df_enhanced.iterrows():
                        try:
                            timestamp = row['Timestamp']
                            date = timestamp.strftime('%Y-%m-%d')
                            time_str = timestamp.strftime('%H:%M')
                            activity = row.get('CatSub', '')

                            # 活動名バリデーション
                            if not activity or pd.isna(activity) or activity == 'unknown':
                                continue

                            # 実測値を取得
                            actual_frustration = row.get('NASA_F')

                            # 生体情報が揃っているかチェック
                            has_biodata = check_fitbit_data_availability(row)

                            # Hourly Logに既に存在するかチェック（キャッシュから）
                            hourly_log = hourly_log_cache.get(date, pd.DataFrame())
                            is_existing = False
                            existing_predicted = None

                            if not hourly_log.empty:
                                existing = hourly_log[
                                    (hourly_log['時刻'] == time_str) &
                                    (hourly_log['活動名'] == activity)
                                ]
                                if not existing.empty:
                                    is_existing = True
                                    existing_row = existing.iloc[0]
                                    existing_predicted = existing_row.get('予測NASA_F')

                            if is_existing:
                                # 予測値が空白で、かつ生体データがある場合は予測値を更新対象に追加
                                if (pd.isna(existing_predicted) or existing_predicted == '') and has_biodata:
                                    update_predictions.append({
                                        'row': row,
                                        'date': date,
                                        'time': time_str,
                                        'activity': activity,
                                        'actual_frustration': actual_frustration
                                    })
                                # それ以外はスキップ（既に登録済み）
                                continue

                            # 新規活動として追加
                            new_activities.append({
                                'row': row,
                                'date': date,
                                'time': time_str,
                                'activity': activity,
                                'actual_frustration': actual_frustration,
                                'has_biodata': has_biodata
                            })

                        except Exception as parse_error:
                            logger.error(f"活動データ解析エラー: {parse_error}")
                            continue

                    logger.warning(f"📊 新規活動: {len(new_activities)}件, 予測値更新: {len(update_predictions)}件")

                    # 予測値更新処理
                    predictions_count = 0
                    for item in update_predictions:
                        try:
                            prediction_result = predictor.predict_from_row(item['row'])
                            if prediction_result and 'predicted_frustration' in prediction_result:
                                predicted_frustration = prediction_result.get('predicted_frustration')
                                if predicted_frustration is not None and not (np.isnan(predicted_frustration) or np.isinf(predicted_frustration)):
                                    predicted_frustration = float(predicted_frustration)
                                    # 予測値を更新
                                    sheets_connector.update_hourly_log_prediction(
                                        user_id, item['date'], item['time'], item['activity'], predicted_frustration
                                    )
                                    predictions_count += 1
                                    logger.warning(f"🔄 予測値更新: {item['activity']} @{item['time']}, 実測={item['actual_frustration']}, 予測={predicted_frustration:.2f}")
                        except Exception as update_error:
                            logger.error(f"予測値更新エラー: {update_error}")
                            continue

                    # 新規活動保存処理
                    for item in new_activities:
                        try:
                            predicted_frustration = None

                            if item['has_biodata']:
                                # 予測実行
                                prediction_result = predictor.predict_from_row(item['row'])
                                if prediction_result and 'predicted_frustration' in prediction_result:
                                    predicted_frustration = prediction_result.get('predicted_frustration')
                                    if predicted_frustration is not None and not (np.isnan(predicted_frustration) or np.isinf(predicted_frustration)):
                                        predicted_frustration = float(predicted_frustration)
                                    else:
                                        predicted_frustration = None

                            # Hourly Logに保存（予測値なしでも保存）
                            hourly_data = {
                                'date': item['date'],
                                'time': item['time'],
                                'activity': item['activity'],
                                'actual_frustration': item['actual_frustration'],
                                'predicted_frustration': predicted_frustration
                            }
                            sheets_connector.save_hourly_log(user_id, hourly_data)
                            predictions_count += 1

                            if predicted_frustration:
                                logger.warning(f"✅ 新規登録: {item['activity']} @{item['time']}, 実測={item['actual_frustration']}, 予測={predicted_frustration:.2f}")
                            else:
                                logger.warning(f"✅ 新規登録: {item['activity']} @{item['time']}, 実測={item['actual_frustration']}, 予測=なし（生体データ不足）")

                        except Exception as save_error:
                            logger.error(f"新規登録エラー: {save_error}")
                            continue

                    logger.warning(f"🎯 処理完了: {user_name}, {predictions_count}件をHourly Logに登録")

                    # last_prediction_resultを更新
                    if predictions_count > 0:
                        last_prediction_result[user_id] = {
                            'timestamp': datetime.now(JST).isoformat(),
                            'user_id': user_id,
                            'user_name': user_name,
                            'predictions_count': predictions_count
                        }

                except Exception as user_error:
                    logger.error(f"{user_name} の処理エラー: {user_error}")
                    continue

        except Exception as e:
            logger.error(f"データ監視ループエラー: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # エラー時も次の実行時刻まで待機
            time.sleep(60)

def run_data_monitor_once():
    """
    データ監視を1回だけ実行（Cloud Scheduler用）
    既存のdata_monitor_loop()のコア処理を1回実行する版
    """
    global last_prediction_result

    logger.warning(f"🔍 データ監視開始（1回実行）: {datetime.now(JST).strftime('%Y-%m-%d %H:%M')}")

    # 全ユーザーのリストをConfigから取得
    users_config = config.get_all_users()

    # 全ユーザーをチェック
    for user_config in users_config:
        user_id = user_config['user_id']
        user_name = user_config['name']

        try:
            # ユーザーごとのpredictorを取得
            predictor = get_predictor(user_id)

            # データ取得（キャッシュなし）
            activity_data = sheets_connector.get_activity_data(user_id, use_cache=False)
            fitbit_data = sheets_connector.get_fitbit_data(user_id, use_cache=False)

            if activity_data.empty:
                logger.warning(f"活動データなし: {user_name}")
                continue

            # nasa_status='done'の行のみフィルタリング
            if 'nasa_status' in activity_data.columns:
                activity_data_done = activity_data[activity_data['nasa_status'] == 'done'].copy()
                logger.warning(f"📊 {user_name}: 全活動={len(activity_data)}件, nasa_status='done'={len(activity_data_done)}件")
            else:
                activity_data_done = activity_data.copy()
                logger.warning(f"⚠️ {user_name}: nasa_status列が見つかりません。全活動を処理します")

            if activity_data_done.empty:
                logger.warning(f"nasa_status='done'の活動なし: {user_name}")
                continue

            # モデル訓練確認
            training_result = ensure_model_trained(user_id, force_retrain=False)
            if training_result['status'] not in ['success', 'already_trained']:
                logger.warning(f"モデル訓練失敗 ({user_name}): {training_result.get('message')}")
                continue

            if predictor.model is None:
                logger.error(f"モデルが初期化されていません ({user_name})")
                continue

            # データ前処理
            activity_processed = predictor.preprocess_activity_data(activity_data_done)
            df_enhanced = predictor.aggregate_fitbit_by_activity(activity_processed, fitbit_data)

            logger.warning(f"🔍 予測チェック開始: {user_name}, 対象活動={len(df_enhanced)}件")

            # 全期間のHourly Logを一度に取得
            all_dates = df_enhanced['Timestamp'].dt.strftime('%Y-%m-%d').unique()
            hourly_log_cache = {}
            for date in all_dates:
                hourly_log_cache[date] = sheets_connector.get_hourly_log(user_id, date)

            logger.warning(f"📋 Hourly Log取得完了: {len(hourly_log_cache)}日分")

            # 新規活動のみを抽出
            new_activities = []
            update_predictions = []

            for idx, row in df_enhanced.iterrows():
                try:
                    timestamp = row['Timestamp']
                    date = timestamp.strftime('%Y-%m-%d')
                    time_str = timestamp.strftime('%H:%M')
                    activity = row.get('CatSub', '')

                    if not activity or pd.isna(activity) or activity == 'unknown':
                        continue

                    actual_frustration = row.get('NASA_F')
                    has_biodata = check_fitbit_data_availability(row)

                    # Hourly Logに既に存在するかチェック
                    hourly_log = hourly_log_cache.get(date, pd.DataFrame())
                    is_existing = False
                    existing_predicted = None

                    if not hourly_log.empty:
                        existing = hourly_log[
                            (hourly_log['時刻'] == time_str) &
                            (hourly_log['活動名'] == activity)
                        ]
                        if not existing.empty:
                            is_existing = True
                            existing_row = existing.iloc[0]
                            existing_predicted = existing_row.get('予測NASA_F')

                    if is_existing:
                        if (pd.isna(existing_predicted) or existing_predicted == '') and has_biodata:
                            update_predictions.append({
                                'row': row,
                                'date': date,
                                'time': time_str,
                                'activity': activity,
                                'actual_frustration': actual_frustration
                            })
                        continue

                    # 新規活動として追加
                    new_activities.append({
                        'row': row,
                        'date': date,
                        'time': time_str,
                        'activity': activity,
                        'actual_frustration': actual_frustration,
                        'has_biodata': has_biodata
                    })

                except Exception as parse_error:
                    logger.error(f"活動データ解析エラー: {parse_error}")
                    continue

            logger.warning(f"📊 新規活動: {len(new_activities)}件, 予測値更新: {len(update_predictions)}件")

            # 予測値更新処理
            predictions_count = 0
            for item in update_predictions:
                try:
                    prediction_result = predictor.predict_from_row(item['row'])
                    if prediction_result and 'predicted_frustration' in prediction_result:
                        predicted_frustration = prediction_result.get('predicted_frustration')
                        if predicted_frustration is not None and not (np.isnan(predicted_frustration) or np.isinf(predicted_frustration)):
                            predicted_frustration = float(predicted_frustration)
                            sheets_connector.update_hourly_log_prediction(
                                user_id, item['date'], item['time'], item['activity'], predicted_frustration
                            )
                            predictions_count += 1
                            logger.warning(f"🔄 予測値更新: {item['activity']} @{item['time']}, 予測={predicted_frustration:.2f}")
                except Exception as update_error:
                    logger.error(f"予測値更新エラー: {update_error}")
                    continue

            # 新規活動保存処理
            for item in new_activities:
                try:
                    predicted_frustration = None

                    if item['has_biodata']:
                        prediction_result = predictor.predict_from_row(item['row'])
                        if prediction_result and 'predicted_frustration' in prediction_result:
                            predicted_frustration = prediction_result.get('predicted_frustration')
                            if predicted_frustration is not None and not (np.isnan(predicted_frustration) or np.isinf(predicted_frustration)):
                                predicted_frustration = float(predicted_frustration)
                            else:
                                predicted_frustration = None

                    hourly_data = {
                        'date': item['date'],
                        'time': item['time'],
                        'activity': item['activity'],
                        'actual_frustration': item['actual_frustration'],
                        'predicted_frustration': predicted_frustration
                    }
                    sheets_connector.save_hourly_log(user_id, hourly_data)
                    predictions_count += 1

                    if predicted_frustration:
                        logger.warning(f"✅ 新規登録: {item['activity']} @{item['time']}, 予測={predicted_frustration:.2f}")
                    else:
                        logger.warning(f"✅ 新規登録: {item['activity']} @{item['time']}, 予測=なし（生体データ不足）")

                except Exception as save_error:
                    logger.error(f"新規登録エラー: {save_error}")
                    continue

            logger.warning(f"🎯 処理完了: {user_name}, {predictions_count}件をHourly Logに登録")

            # last_prediction_resultを更新
            if predictions_count > 0:
                last_prediction_result[user_id] = {
                    'timestamp': datetime.now(JST).isoformat(),
                    'user_id': user_id,
                    'user_name': user_name,
                    'predictions_count': predictions_count
                }

        except Exception as user_error:
            logger.error(f"{user_name} の処理エラー: {user_error}")
            continue

    logger.warning(f"✅ データ監視完了（1回実行）")
    return {'status': 'success', 'processed_users': len(users_config)}

def initialize_application():
    """アプリケーション初期化（一度だけ実行）"""
    global data_monitor_thread, data_monitor_running, user_predictors, _app_initialized

    # 既に初期化済みの場合はスキップ
    if _app_initialized:
        return

    try:
        # 初期化開始ログ（常に出力）
        logger.warning("🚀 アプリケーションを初期化しています...")

        # 既存のモデルをクリア（KNOWN_ACTIVITIESの更新などに対応）
        if user_predictors:
            logger.warning("🔄 既存のモデルをクリアしました。次回アクセス時に新しいKNOWN_ACTIVITIESで再訓練されます。")
            user_predictors.clear()

        # スケジューラー開始
        scheduler.start_scheduler()
        logger.warning("✅ 定期フィードバックスケジューラーを開始しました（毎日14:55 UTC = 23:55 JSTにDiCE実行 + フィードバック生成）")

        # データ更新監視スレッド開始
        data_monitor_running = True
        data_monitor_thread = threading.Thread(target=data_monitor_loop, daemon=True)
        data_monitor_thread.start()
        logger.warning("✅ データ更新監視スレッドを開始しました (15分ごとにチェック)")

        _app_initialized = True
        logger.warning("🎉 アプリケーション初期化完了")
    except Exception as e:
        logger.error(f"アプリケーション初期化エラー: {e}")

@app.route('/api/sheets/recreate-prediction', methods=['POST'])
def recreate_prediction_sheet_endpoint():
    """
    PREDICTION_DATAシートを再作成するAPIエンドポイント
    """
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default')

        logger.info(f"PREDICTION_DATAシート再作成リクエスト: user_id={user_id}")

        # シートを再作成
        result = sheets_connector.recreate_prediction_sheet(user_id)

        if result:
            return jsonify({
                'status': 'success',
                'message': f'PREDICTION_DATA_{user_id}シートを再作成しました',
                'user_id': user_id
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'message': 'シート再作成に失敗しました',
                'user_id': user_id
            }), 500

    except Exception as e:
        logger.error(f"PREDICTION_DATAシート再作成エラー: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/sheets/recreate-daily-summary', methods=['POST'])
def recreate_daily_summary_sheet_endpoint():
    """
    DAILY_SUMMARYシートを再作成するAPIエンドポイント
    """
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default')

        logger.info(f"DAILY_SUMMARYシート再作成リクエスト: user_id={user_id}")

        # シートを再作成
        result = sheets_connector.recreate_daily_summary_sheet(user_id)

        if result:
            return jsonify({
                'status': 'success',
                'message': f'DAILY_SUMMARY_{user_id}シートを再作成しました',
                'user_id': user_id
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'message': 'シート再作成に失敗しました',
                'user_id': user_id
            }), 500

    except Exception as e:
        logger.error(f"DAILY_SUMMARYシート再作成エラー: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def cleanup_application():
    """アプリケーション終了処理"""
    global data_monitor_running

    try:
        if config.ENABLE_INFO_LOGS:
            logger.info("アプリケーションを終了しています...")

        # データ監視スレッド停止
        data_monitor_running = False

        # スケジューラー停止
        scheduler.stop_scheduler()
        if config.ENABLE_INFO_LOGS:
            logger.info("アプリケーション終了完了")
    except Exception as e:
        logger.error(f"アプリケーション終了エラー: {e}")

# Gunicorn/Cloud Run用: モジュールインポート時に初期化
# __name__ == '__main__' の外で実行されるため、本番環境でも動作する
initialize_application()

if __name__ == '__main__':
    try:
        # 開発サーバー起動（本番ではGunicornが使われるのでこのブロックは実行されない）
        port = int(os.environ.get('PORT', 8080))
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

    except KeyboardInterrupt:
        if config.ENABLE_INFO_LOGS:
            logger.info("キーボード割り込みによる終了")
    except Exception as e:
        logger.error(f"アプリケーション実行エラー: {e}")
    finally:
        cleanup_application()