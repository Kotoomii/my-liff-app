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
        DiCEライブラリを使用した反実仮想説明生成 (1日分のデータに対して実行)
        21:00などの定時実行を想定し、その日1日の全活動に対してDiCE提案を生成
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

            # その日1日分のデータに対してDiCEを実行
            target_date = target_timestamp.date()
            logger.info(f"DiCE: {target_date}の1日分のデータに対してDiCE分析を実行します")

            # generate_hourly_alternativesを使用して1日分のDiCE提案を生成
            daily_result = self.generate_hourly_alternatives(df_enhanced, predictor, target_date)

            if daily_result.get('type') == 'hourly_dice_schedule' and daily_result.get('hourly_schedule'):
                # 時間別の提案をフラットな形式に変換
                timeline = []
                for item in daily_result['hourly_schedule']:
                    # 時刻情報を追加
                    hour = item['hour']
                    timestamp = datetime.combine(target_date, datetime.min.time()) + timedelta(hours=hour)
                    timeline.append({
                        'hour': hour,
                        'timestamp': timestamp.isoformat(),
                        'original_timestamp': timestamp.isoformat(),
                        'time_range': item['time_range'],
                        'original_activity': item['original_activity'],
                        'suggested_activity': item['suggested_activity'],
                        'frustration_reduction': item['improvement'],
                        'improvement': item['improvement'],
                        'confidence': item['confidence']
                    })

                return {
                    'type': 'daily_dice_analysis',
                    'date': target_date.strftime('%Y-%m-%d'),
                    'timeline': timeline,
                    'total_improvement': daily_result['total_improvement'],
                    'average_improvement': daily_result.get('average_improvement', 0),
                    'schedule_items': len(timeline),
                    'message': daily_result.get('message', ''),
                    'summary': daily_result.get('summary', ''),
                    'confidence': daily_result.get('confidence', 0.5)
                }
            else:
                return self._get_error_explanation(daily_result.get('error_message', 'DiCE生成に失敗しました'))

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
            # ===== デバッグ開始: 値の出所を追跡 =====
            logger.info(f"DiCE: activity_idx = {activity_idx}")
            logger.info(f"DiCE: 対象行のインデックス = {df_enhanced.index[activity_idx]}")

            # 対象行の全データを確認
            target_row = df_enhanced.iloc[activity_idx]
            logger.info(f"DiCE: 対象行のCatSub = {target_row.get('CatSub', 'N/A')}")
            logger.info(f"DiCE: 対象行のTimestamp = {target_row.get('Timestamp', 'N/A')}")

            # NASA_F値が存在するか確認（実測値）
            if 'NASA_F' in target_row.index:
                logger.info(f"DiCE: 対象行のNASA_F（実測値）= {target_row['NASA_F']}")
            else:
                logger.info("DiCE: 対象行にNASA_F列は存在しません")

            if 'NASA_F_scaled' in target_row.index:
                logger.info(f"DiCE: 対象行のNASA_F_scaled（実測値スケール済み）= {target_row['NASA_F_scaled']}")
            else:
                logger.info("DiCE: 対象行にNASA_F_scaled列は存在しません")

            # 現在の活動のフラストレーション値をモデルで予測
            # クエリインスタンスを準備
            query_features = df_enhanced.iloc[[activity_idx]][predictor.feature_columns]

            # デバッグ: query_featuresの内容を確認
            logger.info(f"DiCE: query_features shape: {query_features.shape}")
            logger.info(f"DiCE: query_features columns: {query_features.columns.tolist()}")
            logger.info(f"DiCE: query_features の主要な値:")
            for col in ['SDNN_scaled', 'Lorenz_Area_scaled', 'hour', 'weekday']:
                if col in query_features.columns:
                    logger.info(f"  - {col} = {query_features[col].iloc[0]}")

            # アクティビティカテゴリを確認
            activity_cols = [col for col in query_features.columns if col.startswith('activity_')]
            active_activity = None
            for col in activity_cols:
                if query_features[col].iloc[0] == 1:
                    active_activity = col.replace('activity_', '')
                    break
            logger.info(f"DiCE: 選択されている活動カテゴリ = {active_activity if active_activity else 'なし'}")

            # 重要な生体情報カラムのNaNチェック
            critical_cols = ['SDNN_scaled', 'Lorenz_Area_scaled']
            for col in critical_cols:
                if col in query_features.columns:
                    val = query_features[col].iloc[0]
                    if pd.isna(val):
                        logger.warning(f"DiCE: {col}がNaNです。履歴データの平均値で補完します")
                        # 履歴データの平均値で補完
                        mean_val = df_enhanced[col].dropna().mean()
                        if pd.notna(mean_val):
                            query_features.loc[query_features.index[0], col] = mean_val
                            logger.info(f"DiCE: {col}を平均値 {mean_val:.4f} で補完しました")
                        else:
                            logger.error(f"DiCE: {col}の平均値も計算できません")
                            return None

            # モデルで予測（スケーリング後の値: 0-1）
            logger.info("DiCE: ===== モデル予測を実行します =====")
            logger.info(f"DiCE: predictor.model のタイプ = {type(predictor.model)}")

            current_frustration_scaled = predictor.model.predict(query_features)[0]

            logger.info(f"DiCE: モデル予測結果（スケール済み 0-1）= {current_frustration_scaled}")

            if np.isnan(current_frustration_scaled) or np.isinf(current_frustration_scaled):
                logger.warning(f"予測値が不正です (NaN/Inf): {current_frustration_scaled}")
                logger.warning(f"query_features値: {query_features.iloc[0].to_dict()}")
                return None

            # スケール戻し（0-1 → 1-20）
            current_frustration = current_frustration_scaled * 20.0

            logger.info(f"DiCE: スケール変換（×20）後のF値 = {current_frustration}")
            logger.info(f"DiCE: ===== モデル予測完了 =====")

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

            # webhooktest.py形式: 生体情報と時間特徴量をcontinuousに指定
            # 曜日はcategoricalとして扱う（continuous_featuresに含めない）
            continuous_features = ['SDNN_scaled', 'Lorenz_Area_scaled', 'hour_sin', 'hour_cos']

            # デバッグ: features_to_varyの内容を確認
            logger.warning(f"🔧 DiCE: predictor.feature_columns数 = {len(predictor.feature_columns)}")
            logger.warning(f"🔧 DiCE: activity_cols（変更可能な列）数 = {len(activity_cols)}")
            logger.warning(f"🔧 DiCE: continuous_features（固定列） = {continuous_features}")
            if len(activity_cols) <= 10:
                logger.warning(f"🔧 DiCE: activity_cols（全て） = {activity_cols}")
            else:
                logger.warning(f"🔧 DiCE: activity_cols数が多いため、最初の10個のみ表示 = {activity_cols[:10]}")

            if len(activity_cols) == 0:
                logger.error("DiCE: activity_colsが空です！活動カテゴリ列が見つかりません")
                return None

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

            # F値は1-20の範囲 → スケーリング後は0.05-1.0
            # desired_rangeを現在のF値に基づいて動的に設定
            # 現在のF値から20-30%程度の改善を目標とする（より現実的な範囲）

            # 改善目標を計算（現在値の20-40%改善）
            improvement_low = max(0.05, current_frustration_scaled * 0.6)   # 40%改善（最小値は0.05）
            improvement_high = max(0.05, current_frustration_scaled * 0.8)  # 20%改善

            # 範囲が狭すぎる場合は最小幅を確保
            if improvement_high - improvement_low < 0.1:
                improvement_low = max(0.05, improvement_high - 0.1)

            desired_range = [improvement_low, improvement_high]

            logger.info(f"DiCE実行: 現在F値(予測値)={current_frustration:.2f}(scaled={current_frustration_scaled:.3f}), 目標範囲={desired_range} (F値{improvement_low*20:.1f}-{improvement_high*20:.1f}に相当)")

            # 生体情報と時間特徴を固定するためのpermitted_range設定
            # features_to_varyで指定されていない列は、元の値から変更されないように制約
            permitted_range = {}
            for col in continuous_features:
                if col in query_features.columns:
                    val = query_features[col].iloc[0]
                    # 生体情報と時間は現在値±0.001の範囲に固定（実質変更不可）
                    permitted_range[col] = [val - 0.001, val + 0.001]

            # 曜日も固定
            weekday_cols = [col for col in query_features.columns if col.startswith('weekday_')]
            for col in weekday_cols:
                val = query_features[col].iloc[0]
                permitted_range[col] = [val, val]  # 完全固定

            logger.warning(f"🔧 DiCE: permitted_range設定 = 生体情報と時間を固定")

            # DiCEで反実仮想例を生成（既に定義したquery_featuresを使用）
            dice_exp = exp.generate_counterfactuals(
                query_instances=query_features,
                total_CFs=5,
                desired_range=desired_range,  # 動的範囲: 現在値から20-40%改善を目標
                features_to_vary=activity_cols,  # 活動カテゴリのみ変更
                permitted_range=permitted_range  # 生体情報・時間を固定
            )

            # 結果を取得
            cf_df = dice_exp.cf_examples_list[0].final_cfs_df

            if cf_df is None or cf_df.empty:
                logger.warning("DiCEが反実仮想例を生成できませんでした")
                return None

            # デバッグ: cf_dfの列を確認
            logger.warning(f"🔍 DiCE cf_df の列: {cf_df.columns.tolist()}")
            logger.warning(f"🔍 DiCE cf_df の行数: {len(cf_df)}")
            if 'NASA_F_scaled' in cf_df.columns:
                logger.warning(f"✅ NASA_F_scaled列が存在します: {cf_df['NASA_F_scaled'].tolist()}")
            else:
                logger.warning(f"❌ NASA_F_scaled列が存在しません！")
                logger.warning(f"   利用可能な列: {[c for c in cf_df.columns if not c.startswith('activity_')][:10]}")

            # デバッグ: 各候補の活動カテゴリを確認
            logger.warning(f"🔍 DiCE cf_df の活動カテゴリ:")
            for i, (idx, cf_row) in enumerate(cf_df.iterrows()):
                active_activities = [col.replace('activity_', '') for col in activity_cols if cf_row[col] == 1]
                logger.warning(f"   候補{i+1}: {active_activities}")

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
                # 最低改善閾値を現在のF値に応じて調整（高F値ほど小さい改善でも許容）
                # F値が高い(15以上)場合: 0.3点以上、中程度(8-15)の場合: 0.5点以上、低い場合(8未満): 1.0点以上
                if current_frustration >= 15:
                    min_improvement = 0.3
                elif current_frustration >= 8:
                    min_improvement = 0.5
                else:
                    min_improvement = 1.0

                if improvement > min_improvement and improvement > best_improvement:
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
                # より詳細なデバッグ情報を追加
                logger.warning(f"DiCE: 有意な改善案が見つかりませんでした")
                logger.warning(f"  - 元の活動: {original_activity_name}")
                logger.warning(f"  - 現在F値(予測値): {current_frustration:.2f} (scaled={current_frustration_scaled:.3f})")
                logger.warning(f"  - 目標範囲: {desired_range} (F値{desired_range[0]*20:.1f}-{desired_range[1]*20:.1f})")
                if cf_df is not None and not cf_df.empty:
                    logger.warning(f"  - DiCEが生成した候補数: {len(cf_df)}件")
                    # 候補の詳細をログ出力
                    for idx, cf_row in cf_df.iterrows():
                        suggested_act = None
                        for act_name in KNOWN_ACTIVITIES:
                            if f'activity_{act_name}' in cf_row.index and cf_row[f'activity_{act_name}'] == 1:
                                suggested_act = act_name
                                break
                        alt_f_scaled = cf_row.get('NASA_F_scaled', 0)
                        alt_f = alt_f_scaled * 20.0
                        imp = current_frustration - alt_f
                        logger.warning(f"    候補{idx+1}: {suggested_act}, F値={alt_f:.2f}, 改善={imp:.2f}点")
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
