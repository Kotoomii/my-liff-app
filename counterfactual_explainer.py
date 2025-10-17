"""
行動単位の反実仮想説明（Counterfactual Explanations）機能
Microsoft Research DiCEライブラリを使用してフラストレーション値改善提案を生成
webhooktest.pyの実装パターンを参考にした正しいDiCE実装
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
import dice_ml
from dice_ml import Dice
from datetime import datetime, timedelta

from config import Config

logger = logging.getLogger(__name__)

# ml_model.pyと同じ活動リスト
KNOWN_ACTIVITIES = [
    '睡眠', '食事', '身のまわりの用事', '療養・静養', '仕事', '仕事のつきあい',
    '授業・学内の活動', '学校外の学習', '炊事・掃除・洗濯', '買い物', '子どもの世話',
    '家庭雑事', '通勤', '通学', '社会参加', '会話・交際', 'スポーツ', '行楽・散策',
    '趣味・娯楽・教養(インターネット除く)', '趣味・娯楽・教養 のインターネット(動画除く)',
    'インターネット動画', 'テレビ', '録画番組・DVD', 'ラジオ', '新聞',
    '雑誌・漫画・本', '音楽', '休息', 'その他', '不明'
]

class ActivityCounterfactualExplainer:
    def __init__(self):
        self.config = Config()

    def generate_activity_based_explanation(self,
                                          df_enhanced: pd.DataFrame,
                                          predictor,
                                          target_timestamp: datetime = None,
                                          lookback_hours: int = 24) -> Dict:
        """
        DiCEライブラリを使用した反実仮想説明生成
        webhooktest.pyの実装パターンに従う
        """
        try:
            if target_timestamp is None:
                target_timestamp = datetime.now()

            if self.config.ENABLE_DEBUG_LOGS:
                logger.debug(f"DiCE説明生成開始: target_timestamp={target_timestamp}")

            # データ存在チェック
            if df_enhanced.empty:
                logger.warning("DiCE説明生成: データが空です")
                return self._get_no_data_explanation("データが空です")

            # モデルチェック
            if predictor is None or predictor.model is None:
                logger.warning("Predictorまたはモデルが初期化されていません")
                return self._get_no_data_explanation("モデルが初期化されていません")

            # 過去24時間の行動変更タイミングを取得
            cutoff_time = target_timestamp - timedelta(hours=lookback_hours)
            recent_activities = df_enhanced[
                (df_enhanced['Timestamp'] >= cutoff_time) &
                (df_enhanced['Timestamp'] <= target_timestamp) &
                (df_enhanced['activity_change'] == 1)
            ].copy()

            if recent_activities.empty:
                logger.info(f"DiCE説明生成: 指定期間に行動変更タイミングがありません")
                return self._get_no_solution_explanation("指定期間に行動変更タイミングがありませんでした")

            # 各行動変更タイミングについてDiCEで反実仮想例を生成
            counterfactual_results = []
            failed_generations = 0

            for idx, activity in recent_activities.iterrows():
                try:
                    cf_result = self._generate_dice_counterfactual(
                        df_enhanced, idx, activity, predictor
                    )
                    if cf_result:
                        counterfactual_results.append(cf_result)
                    else:
                        failed_generations += 1
                except Exception as cf_error:
                    failed_generations += 1
                    logger.warning(f"反実仮想例生成中にエラー: {cf_error}")

            logger.info(f"DiCE生成結果: 成功={len(counterfactual_results)}, 失敗={failed_generations}")

            if not counterfactual_results:
                logger.warning(f"DiCE説明生成: すべての活動で代替案を生成できませんでした")
                return self._get_no_solution_explanation(
                    f"{len(recent_activities)}個の活動を分析しましたが、改善提案を生成できませんでした"
                )

            # 結果をまとめて返す
            return self._summarize_counterfactual_results(counterfactual_results)

        except Exception as e:
            logger.error(f"反実仮想説明生成で予期しないエラー: {e}", exc_info=True)
            return self._get_error_explanation(str(e))

    def _generate_dice_counterfactual(self,
                                     df_enhanced: pd.DataFrame,
                                     activity_idx: int,
                                     activity: pd.Series,
                                     predictor) -> Optional[Dict]:
        """
        DiCEライブラリを使用して単一活動の反実仮想例を生成
        webhooktest.pyのパターンに従う
        """
        try:
            # 訓練データを準備（activity_idxより前のデータ）
            train_data = df_enhanced.iloc[:activity_idx].copy()

            if len(train_data) < 20:
                if self.config.ENABLE_DEBUG_LOGS:
                    logger.debug(f"訓練データが不足（{len(train_data)}件）")
                return None

            # 活動列の準備
            # CatSubカラムが存在する場合、One-Hotエンコーディング
            activity_cols = []
            if 'CatSub' in train_data.columns:
                # CatSubをOne-Hotエンコーディング
                for act in KNOWN_ACTIVITIES:
                    train_data[act] = (train_data['CatSub'] == act).astype(int)
                activity_cols = KNOWN_ACTIVITIES
            else:
                # 既存の活動列を使用
                activity_cols = [col for col in KNOWN_ACTIVITIES if col in train_data.columns]

            if not activity_cols:
                if self.config.ENABLE_DEBUG_LOGS:
                    logger.debug("活動列が見つかりません")
                return None

            # 生体情報列（webhooktest.py形式）
            bio_cols = []
            if 'lorenz_mean' in train_data.columns:
                bio_cols.append('lorenz_mean')
            if 'lorenz_std' in train_data.columns:
                bio_cols.append('lorenz_std')

            if not bio_cols:
                if self.config.ENABLE_DEBUG_LOGS:
                    logger.debug("生体情報列が見つかりません")
                return None

            # 時間特徴量（webhooktest.py形式）
            time_features = []
            if 'hour_sin' in train_data.columns:
                time_features.append('hour_sin')
            if 'hour_cos' in train_data.columns:
                time_features.append('hour_cos')
            weekday_cols = [col for col in train_data.columns if col.startswith('weekday_')]
            time_features.extend(weekday_cols)

            # フラストレーション値のスケーリング
            if 'NASA_F_scaled' not in train_data.columns and 'NASA_F' in train_data.columns:
                train_data['NASA_F_scaled'] = train_data['NASA_F'] / 20.0

            # 特徴量とターゲットの準備
            feature_cols = bio_cols + activity_cols + time_features
            target_col = 'NASA_F_scaled'

            # 欠損値除去
            required_cols = feature_cols + [target_col]
            missing_cols = [col for col in required_cols if col not in train_data.columns]
            if missing_cols:
                if self.config.ENABLE_DEBUG_LOGS:
                    logger.debug(f"必要な列が見つかりません: {missing_cols}")
                return None

            train_clean = train_data.dropna(subset=required_cols)

            if len(train_clean) < 20:
                if self.config.ENABLE_DEBUG_LOGS:
                    logger.debug(f"クリーンな訓練データが不足（{len(train_clean)}件）")
                return None

            # X, y分割（webhooktest.py形式）
            X = train_clean[feature_cols]
            y = train_clean[target_col]

            # DiCEデータオブジェクトを作成
            continuous_features = bio_cols + time_features
            d = dice_ml.Data(
                dataframe=pd.concat([X, y], axis=1),
                continuous_features=continuous_features,
                outcome_name=target_col
            )

            # DiCEモデルオブジェクトを作成
            m = dice_ml.Model(
                model=predictor.model,
                backend="sklearn",
                model_type="regressor"
            )

            # DiCE Explainerを作成
            exp = Dice(d, m)

            # 現在の活動インスタンスを準備
            current_instance = df_enhanced.iloc[[activity_idx]].copy()

            # 現在のインスタンスに必要な列を追加
            for col in feature_cols:
                if col not in current_instance.columns:
                    if col in KNOWN_ACTIVITIES:
                        # 活動列の場合、CatSubから判定
                        current_cat_sub = activity.get('CatSub', '')
                        current_instance[col] = 1 if col == current_cat_sub else 0
                    else:
                        current_instance[col] = 0

            query_instance = current_instance[feature_cols]

            # 現在のフラストレーション値
            current_frustration = activity.get('NASA_F', 10)
            current_frustration_scaled = current_frustration / 20.0

            # DiCEで反実仮想例を生成（webhooktest.py形式）
            # desired_rangeを現在値より低く設定
            desired_max = max(0.0, current_frustration_scaled - 0.05)

            dice_exp = exp.generate_counterfactuals(
                query_instances=query_instance,
                total_CFs=3,
                desired_range=[0.0, desired_max],
                features_to_vary=activity_cols  # 活動のみ変更
            )

            # 結果を取得
            cf_df = dice_exp.cf_examples_list[0].final_cfs_df

            if cf_df is None or cf_df.empty:
                if self.config.ENABLE_DEBUG_LOGS:
                    logger.debug("DiCEが反実仮想例を生成できませんでした")
                return None

            # 最良の反実仮想例を選択
            original = query_instance.iloc[0]
            cf_row = cf_df.iloc[0]

            # 変更された活動を特定（webhooktest.py形式）
            recommended_actions = []
            for col in activity_cols:
                orig_val = original.get(col, 0)
                cf_val = cf_row.get(col, 0)
                if orig_val == 0 and cf_val == 1:
                    recommended_actions.append(col)

            # 改善効果を計算
            alternative_frustration_scaled = cf_row.get(target_col, current_frustration_scaled)
            alternative_frustration = alternative_frustration_scaled * 20.0
            improvement = current_frustration - alternative_frustration

            if improvement > 0 and recommended_actions:
                return {
                    'original_activity': activity.get('CatSub', 'unknown'),
                    'alternative_activity': ', '.join(recommended_actions),
                    'original_frustration': current_frustration,
                    'alternative_frustration': alternative_frustration,
                    'improvement': improvement,
                    'timestamp': activity['Timestamp'],
                    'duration': activity.get('Duration', 0),
                    'confidence': min(0.9, 0.6 + 0.3 * (improvement / 6))
                }

            return None

        except Exception as e:
            logger.error(f"DiCE反実仮想例生成エラー: {e}")
            if self.config.ENABLE_DEBUG_LOGS:
                import traceback
                logger.debug(f"詳細エラー: {traceback.format_exc()}")
            return None

    def _summarize_counterfactual_results(self, counterfactual_results: List[Dict]) -> Dict:
        """反実仮想結果をまとめて説明を生成"""
        try:
            if not counterfactual_results:
                return self._get_fallback_explanation()

            # 改善効果順にソート
            counterfactual_results.sort(key=lambda x: x['improvement'], reverse=True)

            total_improvement = sum(cf['improvement'] for cf in counterfactual_results)
            avg_improvement = total_improvement / len(counterfactual_results)

            # 時間軸での結果を整理
            timeline_results = []
            for cf in counterfactual_results:
                timeline_results.append({
                    'timestamp': cf['timestamp'],
                    'hour': cf['timestamp'].hour,
                    'original_activity': cf['original_activity'],
                    'suggested_activity': cf['alternative_activity'],
                    'frustration_reduction': cf['improvement'],
                    'original_frustration': cf['original_frustration'],
                    'predicted_frustration': cf['alternative_frustration']
                })

            timeline_results.sort(key=lambda x: x['timestamp'])

            # 主要な提案をまとめる
            top_suggestions = counterfactual_results[:3]
            suggestions = []
            for cf in top_suggestions:
                suggestion = f"{cf['timestamp'].strftime('%H:%M')} - " \
                           f"「{cf['original_activity']}」を「{cf['alternative_activity']}」に変更すると" \
                           f"フラストレーション値が{cf['improvement']:.1f}点改善"
                suggestions.append(suggestion)

            result = {
                'type': 'activity_counterfactual',
                'total_improvement': total_improvement,
                'average_improvement': avg_improvement,
                'num_suggestions': len(counterfactual_results),
                'timeline': timeline_results,
                'top_suggestions': suggestions,
                'confidence': min(0.9, 0.6 + 0.3 * len(counterfactual_results) / 10),
                'summary': f"過去24時間で{len(counterfactual_results)}個の行動変更により、"
                          f"平均{avg_improvement:.1f}点のフラストレーション改善が期待できます。"
            }

            return result

        except Exception as e:
            logger.error(f"反実仮想結果まとめエラー: {e}")
            return self._get_fallback_explanation()

    def generate_hourly_alternatives(self, activities_data: pd.DataFrame,
                                   predictor, target_date: datetime = None) -> dict:
        """
        1日の終わりに時間単位の粒度でDiCE改善提案を生成
        """
        try:
            if target_date is None:
                target_date = datetime.now().date()

            if self.config.ENABLE_DEBUG_LOGS:
                logger.debug(f"時間別DiCE提案生成開始: target_date={target_date}")

            # データ存在チェック
            if activities_data.empty:
                logger.warning("時間別DiCE提案生成: 活動データが空です")
                return self._get_no_data_hourly_schedule("活動データが空です")

            # 指定日のデータを抽出
            day_data = activities_data[
                activities_data['Timestamp'].dt.date == target_date
            ].copy()

            if day_data.empty:
                logger.info(f"時間別DiCE提案生成: {target_date}のデータが見つかりません")
                return self._get_no_data_hourly_schedule(f"{target_date}のデータが見つかりませんでした")

            logger.info(f"対象日の活動データ数: {len(day_data)}")

            # 時間別の改善提案を生成
            hourly_schedule = []
            total_improvement = 0
            generation_errors = 0

            # 24時間分のスケジュールを生成
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

                    try:
                        # DiCEを使った代替活動の提案を生成
                        idx = activities_data[activities_data['Timestamp'] == original_activity['Timestamp']].index
                        if len(idx) > 0:
                            result = self._generate_dice_counterfactual(
                                activities_data, idx[0], original_activity, predictor
                            )

                            if result:
                                hourly_schedule.append({
                                    'hour': hour,
                                    'time_range': f"{hour:02d}:00-{hour+1:02d}:00",
                                    'original_activity': result['original_activity'],
                                    'suggested_activity': result['alternative_activity'],
                                    'improvement': result['improvement'],
                                    'confidence': result['confidence']
                                })
                                total_improvement += result['improvement']
                    except Exception as hour_error:
                        generation_errors += 1
                        logger.warning(f"{hour}時台の代替活動生成中にエラー: {hour_error}")

            logger.info(f"時間別DiCE生成結果: 提案数={len(hourly_schedule)}, 合計改善={total_improvement:.1f}")

            if total_improvement == 0:
                logger.warning(f"時間別DiCE提案生成: 改善提案を生成できませんでした")
                return self._get_no_solution_hourly_schedule(
                    f"{len(day_data)}個の活動を分析しましたが、改善提案を生成できませんでした"
                )

            result = {
                'type': 'hourly_dice_schedule',
                'date': target_date.strftime('%Y-%m-%d'),
                'hourly_schedule': hourly_schedule,
                'total_improvement': total_improvement,
                'average_improvement': total_improvement / 24 if hourly_schedule else 0,
                'message': f"今日このような活動をしていたらストレスレベルが{total_improvement:.1f}点下がっていました",
                'confidence': min(0.9, 0.5 + len(hourly_schedule) * 0.05),
                'summary': f"24時間中{len(hourly_schedule)}時間で改善の可能性がありました"
            }

            return result

        except Exception as e:
            logger.error(f"時間別DiCE提案生成で予期しないエラー: {e}", exc_info=True)
            return self._get_error_hourly_schedule(str(e))

    def _get_error_explanation(self, error_message: str) -> Dict:
        """エラー発生時の説明"""
        return {
            'type': 'error',
            'status': 'error',
            'error_message': error_message,
            'total_improvement': 0,
            'timeline': [],
            'confidence': 0.0,
            'user_message': 'DiCEの実行中にエラーが発生しました。'
        }

    def _get_no_data_explanation(self, reason: str) -> Dict:
        """データが不足している場合の説明"""
        return {
            'type': 'no_data',
            'status': 'no_data',
            'reason': reason,
            'total_improvement': 0,
            'timeline': [],
            'confidence': 0.1,
            'user_message': 'データが不足しているため、DiCEによる改善提案を生成できませんでした。'
        }

    def _get_no_solution_explanation(self, reason: str) -> Dict:
        """改善提案が見つからない場合の説明"""
        return {
            'type': 'no_solution',
            'status': 'no_solution',
            'reason': reason,
            'total_improvement': 0,
            'timeline': [],
            'confidence': 0.3,
            'user_message': 'データを分析しましたが、現時点では大きな改善提案を生成できませんでした。'
        }

    def _get_fallback_explanation(self) -> Dict:
        """フォールバック用の基本的な説明"""
        return self._get_no_solution_explanation("代替案を生成できませんでした")

    def _get_error_hourly_schedule(self, error_message: str) -> dict:
        """エラー発生時の時間別スケジュール"""
        return {
            'type': 'error',
            'status': 'error',
            'error_message': error_message,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'hourly_schedule': [],
            'total_improvement': 0,
            'confidence': 0.0,
            'user_message': 'DiCEの実行中にエラーが発生しました。'
        }

    def _get_no_data_hourly_schedule(self, reason: str) -> dict:
        """データが不足している場合の時間別スケジュール"""
        return {
            'type': 'no_data',
            'status': 'no_data',
            'reason': reason,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'hourly_schedule': [],
            'total_improvement': 0,
            'confidence': 0.1,
            'user_message': 'データが不足しているため、DiCEによる改善提案を生成できませんでした。'
        }

    def _get_no_solution_hourly_schedule(self, reason: str) -> dict:
        """改善提案が見つからない場合の時間別スケジュール"""
        return {
            'type': 'no_solution',
            'status': 'no_solution',
            'reason': reason,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'hourly_schedule': [],
            'total_improvement': 0,
            'confidence': 0.3,
            'user_message': 'データを分析しましたが、現時点では大きな改善提案を生成できませんでした。'
        }
