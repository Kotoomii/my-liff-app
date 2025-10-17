"""
行動単位の反実仮想説明（Counterfactual Explanations）機能
webhooktest.pyの成功パターンに基づくDiCE実装
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
import dice_ml
from dice_ml import Dice
from datetime import datetime, timedelta

from config import Config
from ml_model import KNOWN_ACTIVITIES

logger = logging.getLogger(__name__)

class ActivityCounterfactualExplainer:
    def __init__(self):
        self.config = Config()

    def generate_activity_based_explanation(self,
                                          df_enhanced: pd.DataFrame,
                                          predictor,
                                          target_timestamp: datetime = None,
                                          lookback_hours: int = 24) -> Dict:
        """
        DiCEライブラリを使用した反実仮想説明生成 (webhooktest.py形式)
        """
        try:
            if target_timestamp is None:
                target_timestamp = datetime.now()

            if df_enhanced.empty:
                logger.warning("DiCE説明生成: データが空です")
                return self._get_error_explanation("データが空です")

            if predictor is None or predictor.model is None:
                logger.warning("モデルが初期化されていません")
                return self._get_error_explanation("モデルが初期化されていません")

            # 最新のデータを取得
            latest_idx = len(df_enhanced) - 1
            latest_activity = df_enhanced.iloc[latest_idx]

            # DiCEで反実仮想例を生成
            dice_result = self._generate_dice_counterfactual_simple(
                df_enhanced, latest_idx, latest_activity, predictor
            )

            if dice_result:
                return {
                    'type': 'activity_counterfactual',
                    'original_activity': dice_result['original_activity'],
                    'original_frustration': dice_result['original_frustration'],
                    'suggested_activity': dice_result['suggested_activity'],
                    'predicted_frustration': dice_result['predicted_frustration'],
                    'improvement': dice_result['improvement'],
                    'confidence': dice_result['confidence'],
                    'timestamp': latest_activity['Timestamp'],
                    'summary': f"「{dice_result['original_activity']}」を「{dice_result['suggested_activity']}」に変更すると、フラストレーション値が{dice_result['improvement']:.1f}点改善する可能性があります。"
                }
            else:
                return self._get_error_explanation("DiCE生成に失敗しました")

        except Exception as e:
            logger.error(f"反実仮想説明生成エラー: {e}", exc_info=True)
            return self._get_error_explanation(str(e))

    def _generate_dice_counterfactual_simple(self,
                                             df_enhanced: pd.DataFrame,
                                             activity_idx: int,
                                             activity: pd.Series,
                                             predictor) -> Optional[Dict]:
        """
        DiCEを使用して反実仮想例を生成 (webhooktest.py形式のシンプルな実装)
        """
        try:
            # 現在のフラストレーション値
            current_frustration_scaled = activity.get('NASA_F_scaled')
            if pd.isna(current_frustration_scaled):
                logger.warning("NASA_F_scaledがNaNです")
                return None

            current_frustration = current_frustration_scaled * 20.0

            # 訓練データを準備: NaN値を除外
            required_cols = ['SDNN_scaled', 'Lorenz_Area_scaled', 'NASA_F_scaled']
            df_train = df_enhanced.dropna(subset=required_cols).copy()

            if len(df_train) < 20:
                logger.warning(f"訓練データが不足（{len(df_train)}件）")
                return None

            # 特徴量とターゲット
            X_train = df_train[predictor.feature_columns]
            y_train = df_train['NASA_F_scaled']

            # DiCEデータオブジェクトを作成
            # 活動カテゴリ列のみを変更可能にする
            activity_cols = [col for col in predictor.feature_columns if col.startswith('activity_')]
            continuous_features = [col for col in predictor.feature_columns if col not in activity_cols]

            dice_data = pd.concat([X_train, y_train], axis=1)
            d = dice_ml.Data(
                dataframe=dice_data,
                continuous_features=continuous_features,
                outcome_name='NASA_F_scaled'
            )

            # DiCEモデルオブジェクトを作成
            m = dice_ml.Model(
                model=predictor.model,
                backend="sklearn",
                model_type="regressor"
            )

            # DiCE Explainerを作成
            exp = Dice(d, m)

            # クエリインスタンスを準備
            query_instance = df_enhanced.iloc[[activity_idx]][predictor.feature_columns]

            # F値は1-20の範囲 → スケーリング後は0.05-1.0
            # desired_range=[0.05, 0.25]は固定値（F値1-5に相当する範囲）
            # この範囲にフラストレーション値が改善されるような活動を提案

            logger.info(f"DiCE実行: 現在F値={current_frustration:.2f}(scaled={current_frustration_scaled:.3f}), 目標範囲=[0.05, 0.25] (F値1-5に相当)")

            # DiCEで反実仮想例を生成
            dice_exp = exp.generate_counterfactuals(
                query_instances=query_instance,
                total_CFs=5,
                desired_range=[0.05, 0.25],  # 固定範囲: F値1-5を目標
                features_to_vary=activity_cols  # 活動カテゴリのみ変更
            )

            # 結果を取得
            cf_df = dice_exp.cf_examples_list[0].final_cfs_df

            if cf_df is None or cf_df.empty:
                logger.warning("DiCEが反実仮想例を生成できませんでした")
                return None

            # 元の活動カテゴリを特定
            original_activity_name = activity.get('CatSub', 'unknown')

            # 全ての反実仮想例を評価し、最良の改善案を選択
            best_result = None
            best_improvement = 0

            for idx, cf_row in cf_df.iterrows():
                # 反実仮想例の活動カテゴリを取得
                suggested_activity_name = None
                for activity_name in KNOWN_ACTIVITIES:
                    col_name = f'activity_{activity_name}'
                    if col_name in cf_row.index and cf_row[col_name] == 1:
                        suggested_activity_name = activity_name
                        break

                if suggested_activity_name is None:
                    suggested_activity_name = 'unknown'

                # 活動が変わっていない場合はスキップ
                if suggested_activity_name == original_activity_name:
                    continue

                # 改善効果を計算
                alternative_frustration_scaled = cf_row.get('NASA_F_scaled', current_frustration_scaled)
                alternative_frustration = alternative_frustration_scaled * 20.0
                improvement = current_frustration - alternative_frustration

                # 改善が見られ、かつ最良の結果の場合に採用
                # 最低0.5点（スケールで0.025）の改善を要求
                if improvement > 0.5 and improvement > best_improvement:
                    best_improvement = improvement
                    best_result = {
                        'original_activity': original_activity_name,
                        'suggested_activity': suggested_activity_name,
                        'original_frustration': current_frustration,
                        'predicted_frustration': alternative_frustration,
                        'improvement': improvement,
                        'confidence': min(0.9, 0.6 + 0.3 * (improvement / 6))
                    }

            if best_result:
                logger.info(f"DiCE成功: {best_result['original_activity']} → {best_result['suggested_activity']} (改善: {best_improvement:.2f}点)")
                return best_result
            else:
                logger.warning(f"DiCE: 有意な改善案が見つかりませんでした（元の活動: {original_activity_name}, 現在F値: {current_frustration:.2f}）")
                return None

        except Exception as e:
            logger.error(f"DiCE反実仮想例生成エラー: {e}", exc_info=True)
            return None

    def generate_hourly_alternatives(self, activities_data: pd.DataFrame,
                                   predictor, target_date: datetime = None) -> dict:
        """
        1日の終わりに時間単位でDiCE改善提案を生成
        """
        try:
            if target_date is None:
                target_date = datetime.now().date()

            if activities_data.empty:
                logger.warning("時間別DiCE提案: 活動データが空です")
                return self._get_error_hourly_schedule("活動データが空です")

            # 指定日のデータを抽出
            day_data = activities_data[
                activities_data['Timestamp'].dt.date == target_date
            ].copy()

            if day_data.empty:
                logger.info(f"時間別DiCE提案: {target_date}のデータが見つかりません")
                return self._get_error_hourly_schedule(f"{target_date}のデータが見つかりません")

            # 時間別の改善提案を生成
            hourly_schedule = []
            total_improvement = 0

            for hour in range(24):
                hour_start = datetime.combine(target_date, datetime.min.time()) + timedelta(hours=hour)
                hour_end = hour_start + timedelta(hours=1)

                # この時間帯の活動データを取得
                hour_activities = day_data[
                    (day_data['Timestamp'] >= hour_start) &
                    (day_data['Timestamp'] < hour_end)
                ]

                if not hour_activities.empty:
                    original_activity = hour_activities.iloc[0]
                    idx = activities_data.index[activities_data['Timestamp'] == original_activity['Timestamp']]

                    if len(idx) > 0:
                        # DiCEを使った代替活動の提案
                        result = self._generate_dice_counterfactual_simple(
                            activities_data, idx[0], original_activity, predictor
                        )

                        if result:
                            hourly_schedule.append({
                                'hour': hour,
                                'time_range': f"{hour:02d}:00-{hour+1:02d}:00",
                                'original_activity': result['original_activity'],
                                'suggested_activity': result['suggested_activity'],
                                'improvement': result['improvement'],
                                'confidence': result['confidence']
                            })
                            total_improvement += result['improvement']

            if total_improvement == 0:
                logger.warning("時間別DiCE提案: 改善提案を生成できませんでした")
                return self._get_error_hourly_schedule("改善提案を生成できませんでした")

            return {
                'type': 'hourly_dice_schedule',
                'date': target_date.strftime('%Y-%m-%d'),
                'hourly_schedule': hourly_schedule,
                'total_improvement': total_improvement,
                'average_improvement': total_improvement / 24 if hourly_schedule else 0,
                'message': f"今日このような活動をしていたらストレスレベルが{total_improvement:.1f}点下がっていました",
                'confidence': min(0.9, 0.5 + len(hourly_schedule) * 0.05),
                'summary': f"24時間中{len(hourly_schedule)}時間で改善の可能性がありました"
            }

        except Exception as e:
            logger.error(f"時間別DiCE提案生成エラー: {e}", exc_info=True)
            return self._get_error_hourly_schedule(str(e))

    def _get_error_explanation(self, error_message: str) -> Dict:
        """エラー発生時の説明"""
        return {
            'type': 'error',
            'status': 'error',
            'error_message': error_message,
            'confidence': 0.0
        }

    def _get_error_hourly_schedule(self, error_message: str) -> dict:
        """エラー発生時の時間別スケジュール"""
        return {
            'type': 'error',
            'status': 'error',
            'error_message': error_message,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'hourly_schedule': [],
            'total_improvement': 0,
            'confidence': 0.0
        }
