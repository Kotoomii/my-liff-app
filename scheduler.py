"""
定期フィードバック機能
1日の終わりと毎朝の2回、定期的にフィードバックを生成・配信
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta, time
import json
import threading
import time as time_module
import schedule
from dataclasses import dataclass
from enum import Enum

from config import Config
from ml_model import FrustrationPredictor
from counterfactual_explainer import ActivityCounterfactualExplainer
from llm_feedback_generator import LLMFeedbackGenerator
from sheets_connector import SheetsConnector

logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    MORNING = "morning"
    EVENING = "evening"

@dataclass
class FeedbackSchedule:
    morning_time: str = "07:30"  # 毎朝7:30
    evening_time: str = "21:00"  # 毎夜21:00
    enabled: bool = True

class FeedbackScheduler:
    def __init__(self):
        self.config = Config()
        self.predictor = FrustrationPredictor()
        self.explainer = ActivityCounterfactualExplainer()
        self.feedback_generator = LLMFeedbackGenerator()
        self.sheets_connector = SheetsConnector()
        
        self.schedule_config = FeedbackSchedule()
        self.running = False
        self.scheduler_thread = None
        self.feedback_history = []  # 生成したフィードバックの履歴
        
    def start_scheduler(self):
        """
        定期フィードバックスケジューラーを開始
        """
        try:
            if self.running:
                logger.warning("スケジューラーは既に実行中です")
                return
            
            # スケジュール設定
            schedule.every().day.at(self.schedule_config.morning_time).do(
                self._execute_morning_feedback
            )
            schedule.every().day.at(self.schedule_config.evening_time).do(
                self._execute_evening_feedback
            )
            
            logger.info(f"定期フィードバックスケジューラーを開始しました")
            logger.info(f"朝のフィードバック: {self.schedule_config.morning_time}")
            logger.info(f"夜のフィードバック: {self.schedule_config.evening_time}")
            
            self.running = True
            
            # バックグラウンドでスケジューラーを実行
            self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
            self.scheduler_thread.start()
            
        except Exception as e:
            logger.error(f"スケジューラー開始エラー: {e}")
    
    def stop_scheduler(self):
        """
        定期フィードバックスケジューラーを停止
        """
        try:
            self.running = False
            schedule.clear()
            
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=5)
            
            logger.info("定期フィードバックスケジューラーを停止しました")
            
        except Exception as e:
            logger.error(f"スケジューラー停止エラー: {e}")
    
    def _run_scheduler(self):
        """
        スケジューラーのメインループ
        """
        while self.running:
            try:
                schedule.run_pending()
                time_module.sleep(60)  # 1分ごとにチェック
            except Exception as e:
                logger.error(f"スケジューラー実行エラー: {e}")
                time_module.sleep(60)
    
    def _execute_morning_feedback(self):
        """
        朝のフィードバックを実行
        """
        try:
            logger.info("朝のフィードバック生成を開始します")
            
            # 昨日のデータを取得・分析
            yesterday_data = self._get_yesterday_data()
            
            # 全ユーザーに対してフィードバックを生成
            users = self._get_active_users()
            
            for user_id in users:
                morning_feedback = self._generate_user_morning_feedback(user_id, yesterday_data)
                
                if morning_feedback:
                    # フィードバックを保存・配信
                    self._save_and_deliver_feedback(user_id, morning_feedback, FeedbackType.MORNING)
                    logger.info(f"ユーザー {user_id} の朝のフィードバックを生成しました")
            
            logger.info("朝のフィードバック生成が完了しました")
            
        except Exception as e:
            logger.error(f"朝のフィードバック実行エラー: {e}")
    
    def _execute_evening_feedback(self):
        """
        夜のフィードバックを実行
        """
        try:
            logger.info("夜のフィードバック生成を開始します")
            
            # 今日のデータを取得・分析
            today_data = self._get_today_data()
            
            # 全ユーザーに対してフィードバックを生成
            users = self._get_active_users()
            
            for user_id in users:
                evening_feedback = self._generate_user_evening_feedback(user_id, today_data)
                
                if evening_feedback:
                    # フィードバックを保存・配信
                    self._save_and_deliver_feedback(user_id, evening_feedback, FeedbackType.EVENING)
                    logger.info(f"ユーザー {user_id} の夜のフィードバックを生成しました")
            
            logger.info("夜のフィードバック生成が完了しました")
            
        except Exception as e:
            logger.error(f"夜のフィードバック実行エラー: {e}")
    
    def _get_yesterday_data(self) -> Dict:
        """
        昨日のデータを取得
        """
        try:
            yesterday = datetime.now() - timedelta(days=1)
            yesterday_str = yesterday.strftime('%Y-%m-%d')
            
            # 活動データとFitbitデータを取得
            activity_data = self.sheets_connector.get_activity_data()
            fitbit_data = self.sheets_connector.get_fitbit_data()
            
            # 昨日のデータにフィルタリング
            if not activity_data.empty and 'Timestamp' in activity_data.columns:
                activity_data['date'] = pd.to_datetime(activity_data['Timestamp']).dt.date
                yesterday_activity = activity_data[activity_data['date'] == yesterday.date()]
            else:
                yesterday_activity = pd.DataFrame()
            
            if not fitbit_data.empty and 'Timestamp' in fitbit_data.columns:
                fitbit_data['date'] = pd.to_datetime(fitbit_data['Timestamp']).dt.date
                yesterday_fitbit = fitbit_data[fitbit_data['date'] == yesterday.date()]
            else:
                yesterday_fitbit = pd.DataFrame()
            
            return {
                'date': yesterday_str,
                'activity_data': yesterday_activity,
                'fitbit_data': yesterday_fitbit
            }
            
        except Exception as e:
            logger.error(f"昨日のデータ取得エラー: {e}")
            return {
                'date': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                'activity_data': pd.DataFrame(),
                'fitbit_data': pd.DataFrame()
            }
    
    def _get_today_data(self) -> Dict:
        """
        今日のデータを取得
        """
        try:
            today = datetime.now()
            today_str = today.strftime('%Y-%m-%d')
            
            # 活動データとFitbitデータを取得
            activity_data = self.sheets_connector.get_activity_data()
            fitbit_data = self.sheets_connector.get_fitbit_data()
            
            # 今日のデータにフィルタリング
            if not activity_data.empty and 'Timestamp' in activity_data.columns:
                activity_data['date'] = pd.to_datetime(activity_data['Timestamp']).dt.date
                today_activity = activity_data[activity_data['date'] == today.date()]
            else:
                today_activity = pd.DataFrame()
            
            if not fitbit_data.empty and 'Timestamp' in fitbit_data.columns:
                fitbit_data['date'] = pd.to_datetime(fitbit_data['Timestamp']).dt.date
                today_fitbit = fitbit_data[fitbit_data['date'] == today.date()]
            else:
                today_fitbit = pd.DataFrame()
            
            return {
                'date': today_str,
                'activity_data': today_activity,
                'fitbit_data': today_fitbit
            }
            
        except Exception as e:
            logger.error(f"今日のデータ取得エラー: {e}")
            return {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'activity_data': pd.DataFrame(),
                'fitbit_data': pd.DataFrame()
            }
    
    def _get_active_users(self) -> List[str]:
        """
        アクティブなユーザー一覧を取得
        """
        try:
            # 簡単な実装：設定ファイルまたはデフォルトユーザー
            return ['default']  # 実際の実装では、データベースやシートから取得
            
        except Exception as e:
            logger.error(f"アクティブユーザー取得エラー: {e}")
            return ['default']
    
    def _generate_user_morning_feedback(self, user_id: str, yesterday_data: Dict) -> Optional[Dict]:
        """
        特定ユーザーの朝のフィードバックを生成
        """
        try:
            activity_data = yesterday_data.get('activity_data', pd.DataFrame())
            fitbit_data = yesterday_data.get('fitbit_data', pd.DataFrame())
            
            if activity_data.empty:
                logger.warning(f"ユーザー {user_id} の昨日の活動データが見つかりません")
                return self._get_fallback_morning_feedback(user_id)
            
            # データを前処理
            activity_processed = self.predictor.preprocess_activity_data(activity_data)
            if activity_processed.empty:
                return self._get_fallback_morning_feedback(user_id)
            
            # Fitbitデータとの統合
            df_enhanced = self.predictor.aggregate_fitbit_by_activity(activity_processed, fitbit_data)
            
            # Walk Forward Validationで学習（必要に応じて）
            if len(df_enhanced) > 10:
                training_results = self.predictor.walk_forward_validation_train(df_enhanced)
                logger.info(f"ユーザー {user_id} のモデル学習完了")
            
            # DiCE分析を実行（昨日の行動について）
            dice_results = []
            yesterday_end = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            dice_explanation = self.explainer.generate_activity_based_explanation(
                df_enhanced, self.predictor, yesterday_end
            )
            
            if dice_explanation.get('type') != 'fallback':
                dice_results.append(dice_explanation)
            
            # LLMフィードバック生成
            morning_briefing = self.feedback_generator.generate_morning_briefing(
                dice_results
            )
            
            return {
                'user_id': user_id,
                'type': 'morning_feedback',
                'generated_at': datetime.now().isoformat(),
                'date': yesterday_data['date'],
                'briefing': morning_briefing,
                'dice_analysis': dice_results,
                'model_performance': training_results if 'training_results' in locals() else {}
            }
            
        except Exception as e:
            logger.error(f"ユーザー {user_id} の朝のフィードバック生成エラー: {e}")
            return self._get_fallback_morning_feedback(user_id)
    
    def _generate_user_evening_feedback(self, user_id: str, today_data: Dict) -> Optional[Dict]:
        """
        特定ユーザーの夜のフィードバックを生成
        """
        try:
            activity_data = today_data.get('activity_data', pd.DataFrame())
            fitbit_data = today_data.get('fitbit_data', pd.DataFrame())
            
            if activity_data.empty:
                logger.warning(f"ユーザー {user_id} の今日の活動データが見つかりません")
                return self._get_fallback_evening_feedback(user_id)
            
            # データを前処理
            activity_processed = self.predictor.preprocess_activity_data(activity_data)
            if activity_processed.empty:
                return self._get_fallback_evening_feedback(user_id)
            
            # Fitbitデータとの統合
            df_enhanced = self.predictor.aggregate_fitbit_by_activity(activity_processed, fitbit_data)
            
            # 今日の行動についてDiCE分析を実行
            dice_results = []
            now = datetime.now()
            
            dice_explanation = self.explainer.generate_activity_based_explanation(
                df_enhanced, self.predictor, now
            )
            
            if dice_explanation.get('type') != 'fallback':
                dice_results.append(dice_explanation)
            
            # 日次サマリーを生成
            daily_summary = self.explainer.generate_daily_summary(
                dice_results, today_data['date']
            )
            
            # LLM夜のサマリー生成
            evening_summary = self.feedback_generator.generate_evening_summary(dice_results)
            
            return {
                'user_id': user_id,
                'type': 'evening_feedback',
                'generated_at': datetime.now().isoformat(),
                'date': today_data['date'],
                'summary': evening_summary,
                'daily_summary': daily_summary,
                'dice_analysis': dice_results
            }
            
        except Exception as e:
            logger.error(f"ユーザー {user_id} の夜のフィードバック生成エラー: {e}")
            return self._get_fallback_evening_feedback(user_id)
    
    def _save_and_deliver_feedback(self, user_id: str, feedback: Dict, feedback_type: FeedbackType):
        """
        フィードバックを保存し、配信
        """
        try:
            # 履歴に追加
            self.feedback_history.append({
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'feedback_type': feedback_type.value,
                'feedback': feedback
            })
            
            # 履歴の制限（最新100件まで保持）
            if len(self.feedback_history) > 100:
                self.feedback_history = self.feedback_history[-100:]
            
            # 実際の実装では、ここでデータベースやファイルに保存
            # また、Webダッシュボード、メール、Push通知等で配信
            self._save_to_file(feedback)
            
            logger.info(f"フィードバックを保存・配信しました: {user_id}, {feedback_type.value}")
            
        except Exception as e:
            logger.error(f"フィードバック保存・配信エラー: {e}")
    
    def _save_to_file(self, feedback: Dict):
        """
        フィードバックをファイルに保存
        """
        try:
            import os
            feedback_dir = 'feedback_history'
            os.makedirs(feedback_dir, exist_ok=True)
            
            filename = f"{feedback_dir}/feedback_{feedback['type']}_{feedback['generated_at'][:10]}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(feedback, f, ensure_ascii=False, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"ファイル保存エラー: {e}")
    
    def _get_fallback_morning_feedback(self, user_id: str) -> Dict:
        """
        フォールバック用朝のフィードバック
        """
        return {
            'user_id': user_id,
            'type': 'morning_feedback',
            'generated_at': datetime.now().isoformat(),
            'briefing': {
                'type': 'morning_briefing',
                'main_feedback': 'おはようございます！今日も健康的で快適な一日を過ごしましょう。',
                'key_recommendations': ['規則正しい生活リズムを心がけましょう'],
                'confidence': 0.3
            },
            'dice_analysis': [],
            'fallback': True
        }
    
    def _get_fallback_evening_feedback(self, user_id: str) -> Dict:
        """
        フォールバック用夜のフィードバック
        """
        return {
            'user_id': user_id,
            'type': 'evening_feedback',
            'generated_at': datetime.now().isoformat(),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'summary': {
                'type': 'evening_summary',
                'main_feedback': '今日もお疲れさまでした。ゆっくり休んで、明日に備えてください。',
                'achievements': ['今日も一日お疲れさまでした'],
                'tomorrow_recommendations': ['明日も健康的な一日を過ごしてください'],
                'confidence': 0.3
            },
            'dice_analysis': [],
            'fallback': True
        }
    
    def get_recent_feedback(self, user_id: str = None, days: int = 7) -> List[Dict]:
        """
        最近のフィードバック履歴を取得
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            filtered_history = []
            for feedback in self.feedback_history:
                feedback_time = datetime.fromisoformat(feedback['timestamp'])
                
                if feedback_time >= cutoff_date:
                    if user_id is None or feedback['user_id'] == user_id:
                        filtered_history.append(feedback)
            
            return sorted(filtered_history, key=lambda x: x['timestamp'], reverse=True)
            
        except Exception as e:
            logger.error(f"最近のフィードバック取得エラー: {e}")
            return []
    
    def update_schedule_config(self, morning_time: str = None, evening_time: str = None, enabled: bool = None):
        """
        スケジュール設定を更新
        """
        try:
            if morning_time is not None:
                self.schedule_config.morning_time = morning_time
            if evening_time is not None:
                self.schedule_config.evening_time = evening_time
            if enabled is not None:
                self.schedule_config.enabled = enabled
            
            # スケジューラーが実行中の場合は再起動
            if self.running:
                self.stop_scheduler()
                time_module.sleep(1)
                if self.schedule_config.enabled:
                    self.start_scheduler()
            
            logger.info(f"スケジュール設定を更新しました: 朝={self.schedule_config.morning_time}, 夜={self.schedule_config.evening_time}")
            
        except Exception as e:
            logger.error(f"スケジュール設定更新エラー: {e}")
    
    def trigger_manual_feedback(self, user_id: str = 'default', feedback_type: str = 'evening') -> Dict:
        """
        手動でフィードバックを実行
        """
        try:
            if feedback_type == 'morning':
                yesterday_data = self._get_yesterday_data()
                feedback = self._generate_user_morning_feedback(user_id, yesterday_data)
            else:
                today_data = self._get_today_data()
                feedback = self._generate_user_evening_feedback(user_id, today_data)
            
            if feedback:
                feedback_enum = FeedbackType.MORNING if feedback_type == 'morning' else FeedbackType.EVENING
                self._save_and_deliver_feedback(user_id, feedback, feedback_enum)
            
            return feedback or {}
            
        except Exception as e:
            logger.error(f"手動フィードバック実行エラー: {e}")
            return {}
    
    def get_status(self) -> Dict:
        """
        スケジューラーの現在の状態を取得
        """
        return {
            'running': self.running,
            'morning_time': self.schedule_config.morning_time,
            'evening_time': self.schedule_config.evening_time,
            'enabled': self.schedule_config.enabled,
            'feedback_history_count': len(self.feedback_history),
            'next_run': schedule.next_run() if self.running else None
        }