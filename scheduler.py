"""
定期フィードバック機能
1日の終わりと毎朝の2回、定期的にフィードバックを生成・配信
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
import json
import threading
import time as time_module
import schedule
from dataclasses import dataclass
from enum import Enum

# 日本標準時（JST）のタイムゾーン
JST = ZoneInfo('Asia/Tokyo')

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
    evening_time: str = "13:10"  # 13:10 UTC（日本時間22:10 JST）で当日データのDiCE実行
                                 # 22:00のdata_monitor_loop実行後、10分のバッファを確保
    enabled: bool = True

class FeedbackScheduler:
    def __init__(self, user_predictors: Dict = None):
        """
        user_predictors: main.pyで管理されている {user_id: FrustrationPredictor} 辞書
                        data_monitor_loopで学習済みのモデルを使用する
        """
        self.config = Config()
        # 重要: user_predictors が空の辞書 {} でも参照を保持する
        self.user_predictors = user_predictors if user_predictors is not None else {}
        self.sheets_connector = SheetsConnector()
        self.explainer = ActivityCounterfactualExplainer()
        self.feedback_generator = LLMFeedbackGenerator(self.sheets_connector)

        self.schedule_config = FeedbackSchedule()
        self.running = False
        self.scheduler_thread = None
        self.feedback_history = []  # 生成したフィードバックの履歴

    def get_predictor(self, user_id: str) -> FrustrationPredictor:
        """
        ユーザーのpredictorを取得（既に学習済みのものを使用）
        """
        logger.warning(f"🔍 user_predictors辞書の状態: keys={list(self.user_predictors.keys())}, id={id(self.user_predictors)}")

        if user_id not in self.user_predictors:
            logger.warning(f"⚠️ ユーザー {user_id} のpredictorが見つかりません。新規作成します。")
            self.user_predictors[user_id] = FrustrationPredictor()
        else:
            logger.warning(f"✅ ユーザー {user_id} のpredictorが見つかりました（学習済み）")

        return self.user_predictors[user_id]
        
    def start_scheduler(self):
        """
        定期フィードバックスケジューラーを開始
        """
        try:
            if self.running:
                logger.warning("スケジューラーは既に実行中です")
                return
            
            # スケジュール設定（夜のフィードバック + DiCE実行のみ）
            schedule.every().day.at(self.schedule_config.evening_time).do(
                self._execute_evening_feedback
            )

            logger.warning(f"📅 定期フィードバックスケジューラーを開始しました")
            logger.warning(f"⏰ 夜のフィードバック + DiCE実行: {self.schedule_config.evening_time} UTC（日本時間22:10 JST）")
            logger.warning(f"🔄 現在のシステム時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}")

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
            logger.warning(f"🌙 夜のフィードバック生成を開始します（システム時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}）")

            # 今日のデータを取得・分析
            today_data = self._get_today_data()
            logger.warning(f"📊 今日のデータ取得完了: {today_data.get('date')}")

            # 全ユーザーに対してフィードバックを生成
            users = self._get_active_users()
            logger.warning(f"👥 対象ユーザー数: {len(users)}")

            for user_id in users:
                logger.warning(f"🔄 ユーザー {user_id} の処理を開始...")
                evening_feedback = self._generate_user_evening_feedback(user_id, today_data)

                if evening_feedback:
                    # フィードバックを保存・配信
                    self._save_and_deliver_feedback(user_id, evening_feedback, FeedbackType.EVENING)
                    logger.warning(f"✅ ユーザー {user_id} の夜のフィードバックを生成しました")
                else:
                    logger.warning(f"⚠️ ユーザー {user_id} のフィードバック生成に失敗しました")

            logger.warning("🎉 夜のフィードバック生成が完了しました")

        except Exception as e:
            logger.error(f"❌ 夜のフィードバック実行エラー: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _get_today_data(self) -> Dict:
        """
        今日のデータを取得（JST基準）
        """
        try:
            today = datetime.now(JST)
            today_str = today.strftime('%Y-%m-%d')
            logger.warning(f"🗓️ 今日の日付: {today_str}（JST基準: {datetime.now(JST).strftime('%Y-%m-%d %H:%M')}）")
            
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
                'date': datetime.now(JST).strftime('%Y-%m-%d'),
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
        data_monitor_loopで既に学習済みのpredictorを使用
        """
        try:
            activity_data = yesterday_data.get('activity_data', pd.DataFrame())
            fitbit_data = yesterday_data.get('fitbit_data', pd.DataFrame())

            if activity_data.empty:
                logger.warning(f"ユーザー {user_id} の昨日の活動データが見つかりません")
                return self._get_fallback_morning_feedback(user_id)

            # 既に学習済みのpredictorを取得
            predictor = self.get_predictor(user_id)

            # データを前処理
            activity_processed = predictor.preprocess_activity_data(activity_data)
            if activity_processed.empty:
                return self._get_fallback_morning_feedback(user_id)

            # Fitbitデータとの統合
            df_enhanced = predictor.aggregate_fitbit_by_activity(activity_processed, fitbit_data)

            # DiCE分析を実行（昨日の行動について）
            dice_results = []
            yesterday_end = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

            dice_explanation = self.explainer.generate_activity_based_explanation(
                df_enhanced, predictor, yesterday_end
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
        data_monitor_loopで既に学習済みのpredictorを使用してDiCE分析を実行
        """
        try:
            target_activity_data = today_data.get('activity_data', pd.DataFrame())
            target_fitbit_data = today_data.get('fitbit_data', pd.DataFrame())
            target_date = today_data.get('date', '')

            if target_activity_data.empty:
                logger.warning(f"ユーザー {user_id} の対象日（{target_date}）の活動データが見つかりません")
                return self._get_fallback_evening_feedback(user_id)

            logger.warning(f"📊 対象日のデータ: {target_date}, 活動={len(target_activity_data)}件")

            # 既に学習済みのpredictorを取得（data_monitor_loopで学習済み）
            predictor = self.get_predictor(user_id)

            # 🔍 デバッグ: predictorとmodelの状態を詳細に確認
            logger.warning(f"🔍 predictor is None: {predictor is None}")
            logger.warning(f"🔍 predictor.model is None: {predictor.model is None if predictor else 'N/A'}")
            if predictor and predictor.model is not None:
                logger.warning(f"✅ 既に学習済みのpredictorを使用します（model type: {type(predictor.model).__name__}）")
            else:
                logger.error(f"❌ predictor.model が None です！モデルが学習されていません。")
                logger.error(f"   data_monitor_loopでモデル学習が実行されていない可能性があります。")
                return self._get_fallback_evening_feedback(user_id)

            # 【重要】DiCEには全期間データを渡す必要がある
            # DiCEは「このデータセットの中から代替活動を探す」ため、
            # 対象日のデータだけでは候補が不足する
            logger.warning(f"📦 DiCE用に全期間データを取得...")
            all_activity_data = self.sheets_connector.get_activity_data(user_id)
            all_fitbit_data = self.sheets_connector.get_fitbit_data(user_id)

            all_activity_processed = predictor.preprocess_activity_data(all_activity_data)
            all_df_enhanced = predictor.aggregate_fitbit_by_activity(all_activity_processed, all_fitbit_data)
            logger.warning(f"📊 全期間データ: {len(all_df_enhanced)}件")

            # 対象日のデータを前処理（DiCE分析対象の日付を指定するため）
            target_activity_processed = predictor.preprocess_activity_data(target_activity_data)
            if target_activity_processed.empty:
                logger.warning(f"⚠️ 対象日のデータ前処理後が空です")
                return self._get_fallback_evening_feedback(user_id)

            target_df_enhanced = predictor.aggregate_fitbit_by_activity(target_activity_processed, target_fitbit_data)
            logger.warning(f"📊 対象日データ: {len(target_df_enhanced)}件")

            # 対象日の行動についてDiCE分析を実行
            # 全期間データを渡し、その中から対象日のデータに対する改善案を生成
            dice_results = []

            # target_dateは文字列なのでdatetimeオブジェクトに変換
            from datetime import datetime as dt_class
            target_datetime = dt_class.strptime(target_date, '%Y-%m-%d')

            logger.warning(f"🎲 DiCE分析を開始します（全期間: {len(all_df_enhanced)}件, 対象日: {target_date}）...")
            logger.warning(f"📝 DiCE結果は1件生成されるたびにHourly Logに書き込みます")

            # コールバック関数：DiCE結果を1件ずつHourly Logに書き込む
            saved_count = [0]  # リストで包んで参照を保持

            def save_dice_result_callback(result):
                """DiCE結果を1件受け取って即座にHourly Logに保存"""
                try:
                    date = target_date
                    time = result.get('time', '')
                    original_activity = result.get('original_activity', '')
                    suggested_activity = result.get('suggested_activity', '')
                    original_f = result.get('original_frustration')
                    improved_f = result.get('predicted_frustration')

                    # 改善幅を計算（正の値が改善を表す）
                    improvement = original_f - improved_f if (original_f and improved_f) else None

                    # ログ出力（improvementがNoneの場合に備える）
                    improvement_str = f"{improvement:.2f}" if improvement is not None else "N/A"
                    logger.warning(f"  💡 {date} {time} {original_activity} → {suggested_activity} (改善: {improvement_str})")

                    # Hourly Logを更新
                    success = self.sheets_connector.update_hourly_log_with_dice(
                        user_id=user_id,
                        date=date,
                        time=time,
                        activity=original_activity,
                        dice_suggestion=suggested_activity,
                        improvement=improvement,
                        improved_frustration=improved_f
                    )

                    if success:
                        saved_count[0] += 1
                        logger.warning(f"  ✅ Hourly Logに保存しました（{saved_count[0]}件目）")

                except Exception as e:
                    logger.error(f"❌ Hourly Log DiCE更新エラー: {e}")

            # DiCE分析を実行（コールバックで1件ずつ保存）
            dice_explanation = self.explainer.generate_activity_based_explanation(
                all_df_enhanced, predictor, target_datetime, callback=save_dice_result_callback
            )

            logger.warning(f"🎲 DiCE分析完了: type={dice_explanation.get('type')}, 保存件数={saved_count[0]}件")

            if dice_explanation.get('type') != 'daily_dice_analysis':
                logger.error(f"❌ DiCE分析が失敗しました（type={dice_explanation.get('type')}）")
                logger.error(f"   エラーメッセージ: {dice_explanation.get('error_message', 'なし')}")

            # Hourly Logから今日のデータを再取得してフィードバック生成
            logger.warning(f"💬 Hourly LogからDiCE提案を含むデータを取得中...")
            hourly_log = self.sheets_connector.get_hourly_log(user_id, today_data['date'])

            # タイムラインデータとDiCE提案を構築
            timeline_data = []
            hourly_schedule = []  # DiCE提案のリスト

            for idx, row in hourly_log.iterrows():
                activity = row.get('活動名')
                time_str = row.get('時刻')
                predicted_f = row.get('予測NASA_F')
                dice_suggestion = row.get('DiCE提案活動名')
                improvement = row.get('改善幅')
                improved_f = row.get('改善後F値')

                # タイムラインデータに追加
                if pd.notna(predicted_f):
                    timeline_data.append({
                        'time': time_str,
                        'activity': activity,
                        'frustration_value': float(predicted_f)
                    })

                # DiCE提案がある場合
                if pd.notna(dice_suggestion) and dice_suggestion != '':
                    # 元の活動データからDurationを取得してtime_rangeを計算
                    time_range = time_str  # デフォルトは時刻のみ
                    duration_minutes = 60  # デフォルト60分

                    try:
                        # target_activity_dataから該当する活動を探す
                        # Timestampの時刻部分とactivity名でマッチング
                        if not target_activity_data.empty and 'Timestamp' in target_activity_data.columns:
                            # 時刻文字列をパース（例: "13:00"）
                            from datetime import datetime as dt_class
                            target_time_obj = dt_class.strptime(f"{target_date} {time_str}", '%Y-%m-%d %H:%M')

                            # 該当する活動を探す
                            matching_activities = target_activity_data[
                                (target_activity_data['Timestamp'].dt.strftime('%H:%M') == time_str) &
                                (target_activity_data['CatSub'] == activity)
                            ]

                            if not matching_activities.empty:
                                # Durationを取得（分単位、numpy.int64をintに変換）
                                duration_minutes = int(matching_activities.iloc[0].get('Duration', 60))

                                # time_rangeを計算（例: "13:00-16:00"）
                                from datetime import timedelta
                                end_time_obj = target_time_obj + timedelta(minutes=duration_minutes)
                                time_range = f"{time_str}-{end_time_obj.strftime('%H:%M')}"
                                logger.info(f"  📐 {activity} の time_range を計算: {time_range} (Duration: {duration_minutes}分)")
                    except Exception as e:
                        logger.warning(f"  ⚠️ time_range計算エラー（デフォルト値を使用）: {e}")

                    hourly_schedule.append({
                        'time': time_str,
                        'time_range': time_range,  # 追加
                        'original_activity': activity,
                        'suggested_activity': dice_suggestion,
                        'improvement': float(improvement) if pd.notna(improvement) else 0,
                        'original_frustration': float(predicted_f) if pd.notna(predicted_f) else None,
                        'improved_frustration': float(improved_f) if pd.notna(improved_f) else None
                    })

            logger.warning(f"📊 タイムラインデータ: {len(timeline_data)}件, DiCE提案: {len(hourly_schedule)}件")

            # DiCE結果を構築
            dice_result = {
                'hourly_schedule': hourly_schedule,
                'total_improvement_potential': sum([s.get('improvement', 0) or 0 for s in hourly_schedule])
            }

            # LLMで日次フィードバックを生成（内部で昨日のデータを取得）
            feedback_result_llm = self.feedback_generator.generate_daily_dice_feedback(
                dice_result,
                timeline_data,
                user_id=user_id
            )
            logger.warning(f"💬 LLMフィードバック生成完了")

            # 日次平均を計算
            predicted_values = [item['frustration_value'] for item in timeline_data]
            avg_predicted = sum(predicted_values) / len(predicted_values) if predicted_values else None

            # 日次平均実測を計算（Activity_DataからNASA_F値を取得）
            avg_actual = None
            if not target_activity_data.empty and 'NASA_F' in target_activity_data.columns:
                actual_values = target_activity_data['NASA_F'].dropna()
                if not actual_values.empty:
                    avg_actual = float(actual_values.mean())
                    logger.warning(f"📊 日次平均実測: {avg_actual:.2f}点 (n={len(actual_values)})")
                else:
                    logger.warning(f"⚠️ NASA_F値がありません（全てNaN）")
            else:
                logger.warning(f"⚠️ Activity_Dataが空、またはNASA_F列がありません")

            # Daily Summaryに保存
            summary_data = {
                'date': today_data['date'],
                'avg_actual': avg_actual,
                'avg_predicted': avg_predicted,
                'dice_improvement': feedback_result_llm.get('total_improvement_potential', 0),
                'dice_count': feedback_result_llm.get('num_suggestions', 0),
                'chatgpt_feedback': feedback_result_llm.get('main_feedback', ''),
                'action_plan': feedback_result_llm.get('action_plan', []),
                'generated_at': feedback_result_llm.get('generated_at', datetime.now().isoformat()),
                'total_activities': len(timeline_data)
            }

            save_success = self.sheets_connector.save_daily_feedback_summary(user_id, summary_data)
            if save_success:
                logger.warning(f"💾 Daily Summary保存完了: user_id={user_id}, date={today_data['date']}")
            else:
                logger.warning(f"⚠️ Daily Summary保存失敗: user_id={user_id}")

            feedback_result = {
                'user_id': user_id,
                'type': 'evening_feedback',
                'generated_at': datetime.now().isoformat(),
                'date': today_data['date'],
                'feedback': feedback_result_llm,
                'daily_stats': {
                    'avg_predicted': round(avg_predicted, 2) if avg_predicted is not None else None,
                    'total_activities': len(timeline_data),
                    'dice_suggestions': len(hourly_schedule)
                },
                'saved_to_spreadsheet': save_success
            }

            logger.warning(f"🎉 ユーザー {user_id} の夜のフィードバック生成完了")
            return feedback_result
            
        except Exception as e:
            logger.error(f"❌ ユーザー {user_id} の夜のフィードバック生成エラー: {e}")
            import traceback
            logger.error(traceback.format_exc())
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