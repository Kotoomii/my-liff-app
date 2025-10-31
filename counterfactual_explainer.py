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
                                          lookback_hours: int = 24,
                                          callback=None) -> Dict:
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
            daily_result = self.generate_hourly_alternatives(df_enhanced, predictor, target_date, callback=callback)

            if daily_result.get('type') == 'hourly_dice_schedule' and daily_result.get('hourly_schedule'):
                # 時間別の提案をフラットな形式に変換
                timeline = []
                for item in daily_result['hourly_schedule']:
                    # 時刻情報を追加
                    hour = item['hour']
                    timestamp = datetime.combine(target_date, datetime.min.time()) + timedelta(hours=hour)
                    timeline.append({
                        'hour': hour,
                        'time': item.get('time'),  # 実際のTimestamp (例: "14:30")
                        'timestamp': timestamp.isoformat(),
                        'original_timestamp': timestamp.isoformat(),
                        'time_range': item['time_range'],
                        'original_activity': item['original_activity'],
                        'suggested_activity': item['suggested_activity'],
                        'original_frustration': item.get('original_frustration'),  # 現在のF値
                        'predicted_frustration': item.get('predicted_frustration'),  # 改善後のF値
                        'frustration_reduction': item['improvement'],
                        'improvement': item['improvement'],
                        'confidence': item['confidence']
                    })

                return {
                    'type': 'daily_dice_analysis',
                    'date': target_date.strftime('%Y-%m-%d'),
                    'timeline': timeline,
                    'hourly_schedule': timeline,  # schedulerとの互換性のため追加
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
            logger.warning(f"🎲 DiCE個別処理開始: activity_idx={activity_idx}")
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
            # 生体情報がない場合はスキップする（平均値で補完しない）
            critical_cols = ['SDNN_scaled', 'Lorenz_Area_scaled']
            for col in critical_cols:
                if col in query_features.columns:
                    val = query_features[col].iloc[0]
                    if pd.isna(val):
                        logger.info(f"DiCE: {col}がNaNです。生体情報がないためこの活動をスキップします")
                        return None
                else:
                    logger.error(f"DiCE: {col}列が見つかりません")
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

            # 訓練データを準備: NaN値と'CatSub'が欠損している行を除外
            required_cols = ['SDNN_scaled', 'Lorenz_Area_scaled', 'NASA_F_scaled', 'CatSub']
            df_train = df_enhanced.dropna(subset=required_cols).copy()

            if len(df_train) < 20:
                logger.warning(f"訓練データが不足（{len(df_train)}件）")
                return None

            # DiCE用のデータフレームを作成: CatSub列と生体情報、時間特徴量のみを含む
            # One-Hotエンコードされた活動列は含めない
            dice_features = ['CatSub', 'SDNN_scaled', 'Lorenz_Area_scaled', 'hour_sin', 'hour_cos']

            # 曜日列を追加
            weekday_cols = [col for col in df_train.columns if col.startswith('weekday_')]
            dice_features.extend(weekday_cols)

            # DiCE用のデータフレームを作成
            df_dice_train = df_train[dice_features + ['NASA_F_scaled']].copy()

            # 'CatSub'をカテゴリカル型に変換（訓練データに存在するカテゴリのみ）
            # カテゴリ一覧を明示的に取得
            train_categories = sorted(df_dice_train['CatSub'].unique().tolist())
            logger.warning(f"🔧 DiCE: 訓練データに存在するCatSub = {train_categories}")

            df_dice_train['CatSub'] = pd.Categorical(df_dice_train['CatSub'], categories=train_categories)

            logger.warning(f"🔧 DiCE: CatSub列をカテゴリカル型に変換しました")
            logger.warning(f"🔧 DiCE: CatSubのカテゴリ数 = {df_dice_train['CatSub'].nunique()}")

            # クエリインスタンスの準備: CatSub列を含める
            query_catsub = target_row.get('CatSub')
            logger.warning(f"🔧 DiCE: 元の活動 (query) = {query_catsub}")

            # クエリのCatSubが訓練データに存在するか確認
            if query_catsub not in train_categories:
                logger.info(f"ℹ️ DiCE: 元の活動 '{query_catsub}' が訓練データに存在しないため、DiCE提案をスキップします")
                logger.debug(f"   訓練データに存在するカテゴリ: {train_categories}")
                return None

            query_dict = {
                'CatSub': [query_catsub],
                'SDNN_scaled': [query_features['SDNN_scaled'].iloc[0]],
                'Lorenz_Area_scaled': [query_features['Lorenz_Area_scaled'].iloc[0]],
                'hour_sin': [query_features['hour_sin'].iloc[0]],
                'hour_cos': [query_features['hour_cos'].iloc[0]]
            }

            # 曜日列をクエリに追加
            for col in weekday_cols:
                query_dict[col] = [query_features[col].iloc[0]]

            query_dice = pd.DataFrame(query_dict)
            # query_diceのCatSubも訓練データと同じカテゴリで設定
            query_dice['CatSub'] = pd.Categorical(query_dice['CatSub'], categories=train_categories)

            logger.warning(f"🔧 DiCE: query_dice = {query_dice.to_dict('records')[0]}")
            logger.warning(f"🔧 DiCE: query CatSubのカテゴリコード = {query_dice['CatSub'].cat.codes[0]}")

            # webhooktest.py形式: 生体情報と時間特徴量をcontinuousに指定
            continuous_features = ['SDNN_scaled', 'Lorenz_Area_scaled', 'hour_sin', 'hour_cos']
            # 曜日列もcontinuousとして扱う（One-Hotエンコード済みのため）
            continuous_features.extend(weekday_cols)

            logger.warning(f"🔧 DiCE: continuous_features = {continuous_features}")
            logger.warning(f"🔧 DiCE: CatSubをカテゴリカル変数として扱います")

            # DiCEデータオブジェクトを作成
            d = dice_ml.Data(
                dataframe=df_dice_train,
                continuous_features=continuous_features,
                outcome_name='NASA_F_scaled'
            )

            # モデルラッパークラスを作成
            # CatSubをOne-Hotエンコーディングしてから元のモデルで予測する
            class ModelWrapper:
                def __init__(self, original_model, feature_columns, known_activities):
                    self.original_model = original_model
                    self.feature_columns = feature_columns
                    self.known_activities = known_activities
                    self.call_count = 0  # 呼び出し回数をカウント

                def predict(self, X):
                    """CatSubをOne-Hotエンコーディングしてから予測"""
                    self.call_count += 1
                    logger.warning(f"🔧🔧🔧 ModelWrapper.predict() 呼び出し #{self.call_count}")
                    logger.warning(f"🔧 入力データの形状: {X.shape}")
                    logger.warning(f"🔧 入力データの列: {X.columns.tolist()}")

                    X_encoded = X.copy()

                    # CatSubをOne-Hotエンコーディング
                    if 'CatSub' in X_encoded.columns:
                        # カテゴリカル型をstr型に変換
                        catsub_values = X_encoded['CatSub'].astype(str)

                        logger.warning(f"🔧 ModelWrapper: CatSub値 = {catsub_values.tolist()}")

                        for activity in self.known_activities:
                            X_encoded[f'activity_{activity}'] = (catsub_values == activity).astype(int)

                        # デバッグ: One-Hotエンコーディングの結果を確認
                        activity_cols = [f'activity_{act}' for act in self.known_activities]
                        active_activities = []
                        for idx in range(len(X_encoded)):
                            row_activities = [col.replace('activity_', '') for col in activity_cols
                                            if col in X_encoded.columns and X_encoded[col].iloc[idx] == 1]
                            active_activities.append(row_activities)
                        logger.warning(f"🔧 ModelWrapper: One-Hot結果 = {active_activities}")

                        # CatSub列を削除
                        X_encoded = X_encoded.drop('CatSub', axis=1)
                    else:
                        logger.error(f"❌ ModelWrapper: CatSub列が見つかりません！列: {X_encoded.columns.tolist()}")

                    # 必要な列のみを選択（順序も元のfeature_columnsに合わせる）
                    try:
                        X_final = X_encoded[self.feature_columns]
                    except KeyError as e:
                        logger.error(f"❌ ModelWrapper: 列が不足しています: {e}")
                        logger.error(f"   必要な列: {self.feature_columns}")
                        logger.error(f"   実際の列: {X_encoded.columns.tolist()}")
                        raise

                    # 予測を実行
                    predictions = self.original_model.predict(X_final)
                    logger.warning(f"🔧 ModelWrapper: 予測結果 = {predictions.tolist()}")
                    logger.warning(f"🔧🔧🔧 ModelWrapper.predict() 完了 #{self.call_count}")

                    return predictions

            # ラッパーモデルを作成
            wrapped_model = ModelWrapper(predictor.model, predictor.feature_columns, KNOWN_ACTIVITIES)

            # DiCEモデルオブジェクトを作成（ラッパーモデルを使用）
            m = dice_ml.Model(
                model=wrapped_model,
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
            for col in ['SDNN_scaled', 'Lorenz_Area_scaled', 'hour_sin', 'hour_cos']:
                if col in query_dice.columns:
                    val = query_dice[col].iloc[0]
                    # 生体情報と時間は現在値±0.001の範囲に固定（実質変更不可）
                    permitted_range[col] = [val - 0.001, val + 0.001]

            # 曜日列も固定
            for col in weekday_cols:
                if col in query_dice.columns:
                    val = query_dice[col].iloc[0]
                    permitted_range[col] = [val - 0.001, val + 0.001]

            logger.warning(f"🔧 DiCE: permitted_range設定 = 生体情報、時間、曜日を固定")
            logger.warning(f"🔧 DiCE: features_to_vary = ['CatSub'] のみ")

            # 🔍 DiCE実行前の最終確認
            logger.warning(f"🔍🔍🔍 DiCE実行前の最終確認")
            logger.warning(f"🔍 訓練データ:")
            logger.warning(f"   - 行数: {len(df_dice_train)}")
            logger.warning(f"   - CatSubユニーク値数: {df_dice_train['CatSub'].nunique()}")
            logger.warning(f"   - CatSubユニーク値: {df_dice_train['CatSub'].unique().tolist()}")
            logger.warning(f"   - NASA_F_scaled 範囲: [{df_dice_train['NASA_F_scaled'].min():.3f}, {df_dice_train['NASA_F_scaled'].max():.3f}]")
            logger.warning(f"   - NASA_F_scaled 平均: {df_dice_train['NASA_F_scaled'].mean():.3f}")
            logger.warning(f"🔍 クエリ:")
            logger.warning(f"   - CatSub: {query_dice['CatSub'].iloc[0]}")
            logger.warning(f"   - 現在のF値(予測): {current_frustration:.2f} (scaled={current_frustration_scaled:.3f})")
            logger.warning(f"🔍 目標:")
            logger.warning(f"   - desired_range: {desired_range}")
            logger.warning(f"   - F値換算: [{desired_range[0]*20:.2f}, {desired_range[1]*20:.2f}]")
            logger.warning(f"   - 改善幅: {(current_frustration_scaled - desired_range[1])*20:.2f} 〜 {(current_frustration_scaled - desired_range[0])*20:.2f} 点")

            # DiCEで反実仮想例を生成（CatSub列を使用したquery_diceを使用）
            logger.warning(f"🚀🚀🚀 DiCE.generate_counterfactuals()を開始します...")
            logger.warning(f"🚀 ModelWrapper呼び出し回数（開始前）: {wrapped_model.call_count}")

            dice_exp = exp.generate_counterfactuals(
                query_instances=query_dice,
                total_CFs=5,
                desired_range=desired_range,  # 動的範囲: 現在値から20-40%改善を目標
                features_to_vary=['CatSub'],  # CatSubのみ変更
                permitted_range=permitted_range  # 生体情報・時間・曜日を固定
            )

            logger.warning(f"🚀🚀🚀 DiCE.generate_counterfactuals()が完了しました")
            logger.warning(f"🚀 ModelWrapper呼び出し回数（完了後）: {wrapped_model.call_count}")

            # 結果を取得
            logger.warning(f"🔍🔍🔍 DiCE生成結果を取得します")
            logger.warning(f"🔍 dice_exp.cf_examples_list の長さ: {len(dice_exp.cf_examples_list)}")

            if len(dice_exp.cf_examples_list) == 0:
                logger.error("❌ DiCE: cf_examples_listが空です！反実仮想例が1つも生成されませんでした")
                logger.error(f"   - 現在のF値: {current_frustration:.2f} (scaled={current_frustration_scaled:.3f})")
                logger.error(f"   - 目標範囲: {desired_range}")
                logger.error(f"   - 訓練データ数: {len(df_dice_train)}")
                logger.error(f"   - CatSubユニーク数: {df_dice_train['CatSub'].nunique()}")
                return None

            cf_df = dice_exp.cf_examples_list[0].final_cfs_df
            logger.warning(f"🔍 cf_df is None: {cf_df is None}")
            logger.warning(f"🔍 cf_df is empty: {cf_df.empty if cf_df is not None else 'N/A'}")

            if cf_df is None:
                logger.error("❌ DiCE: final_cfs_dfがNoneです！")
                return None

            if cf_df.empty:
                logger.error("❌ DiCE: final_cfs_dfが空です！反実仮想例を生成できませんでした")
                logger.error(f"   - 現在のF値: {current_frustration:.2f} (scaled={current_frustration_scaled:.3f})")
                logger.error(f"   - 目標範囲: {desired_range} (F値{desired_range[0]*20:.1f}-{desired_range[1]*20:.1f})")
                logger.error(f"   - 制約条件が厳しすぎる可能性があります")
                return None

            # デバッグ: cf_dfの列を確認
            logger.warning(f"🔍 DiCE cf_df の列: {cf_df.columns.tolist()}")
            logger.warning(f"🔍 DiCE cf_df の行数: {len(cf_df)}")
            if 'NASA_F_scaled' in cf_df.columns:
                logger.warning(f"⚠️  DiCEが返したNASA_F_scaled: {cf_df['NASA_F_scaled'].tolist()}")
            else:
                logger.warning(f"❌ NASA_F_scaled列が存在しません！")
                logger.warning(f"   利用可能な列: {cf_df.columns.tolist()}")

            # DiCEが返したNASA_F_scaledは信頼できないため、ModelWrapperで明示的に予測し直す
            logger.warning(f"🔧 ModelWrapperで明示的にNASA_F_scaledを予測し直します")

            # 🔍 DiCEが生成したCatSubの値を詳細に確認
            logger.warning(f"🔍🔍🔍 DiCE生成後のCatSub詳細チェック開始")
            logger.warning(f"🔍 cf_df['CatSub']のdtype: {cf_df['CatSub'].dtype}")
            logger.warning(f"🔍 cf_df['CatSub']の値: {cf_df['CatSub'].tolist()}")
            logger.warning(f"🔍 cf_df['CatSub']の型: {[type(x) for x in cf_df['CatSub'].tolist()]}")

            # カテゴリカル型の場合、カテゴリ名に変換
            if pd.api.types.is_categorical_dtype(cf_df['CatSub']):
                logger.warning(f"🔍 CatSubがカテゴリカル型です！カテゴリ名に変換します")
                logger.warning(f"🔍 カテゴリコード: {cf_df['CatSub'].cat.codes.tolist()}")
                logger.warning(f"🔍 カテゴリ一覧: {cf_df['CatSub'].cat.categories.tolist()}")
                # カテゴリコードをカテゴリ名に変換
                cf_df['CatSub'] = cf_df['CatSub'].astype(str)
                logger.warning(f"🔍 変換後のCatSub: {cf_df['CatSub'].tolist()}")

            # cf_dfからNASA_F_scaled列を削除
            cf_features_only = cf_df.drop('NASA_F_scaled', axis=1, errors='ignore').copy()

            # ModelWrapperで予測
            logger.warning(f"🔧 ModelWrapper.predict()を呼び出します...")
            logger.warning(f"🔧 予測に使用するCatSub: {cf_features_only['CatSub'].tolist()}")
            predicted_f_values = wrapped_model.predict(cf_features_only)
            logger.warning(f"🔧 ModelWrapperの予測結果: {predicted_f_values.tolist()}")

            # 予測結果で上書き
            cf_df['NASA_F_scaled'] = predicted_f_values
            logger.warning(f"✅ 上書き後のNASA_F_scaled: {cf_df['NASA_F_scaled'].tolist()}")

            # 🔍 各候補のCatSubとF値の対応を確認
            logger.warning(f"🔍🔍🔍 各候補の活動とF値の対応確認:")
            for i, (idx, cf_row) in enumerate(cf_df.iterrows()):
                cf_activity = cf_row.get('CatSub')
                cf_f = cf_row.get('NASA_F_scaled')
                logger.warning(f"🔍   候補{i+1}: CatSub='{cf_activity}' → F_scaled={cf_f:.4f} (F値={cf_f*20:.2f})")

            # デバッグ: 各候補の活動カテゴリと生体情報を確認
            logger.warning(f"🔍 DiCE cf_df の活動カテゴリと生体情報:")
            for i, (idx, cf_row) in enumerate(cf_df.iterrows()):
                activity_name = cf_row.get('CatSub', 'N/A')
                sdnn = cf_row.get('SDNN_scaled', 'N/A')
                lorenz = cf_row.get('Lorenz_Area_scaled', 'N/A')
                f_scaled = cf_row.get('NASA_F_scaled', 'N/A')
                # フォーマット指定子を条件式の外で適用
                sdnn_str = f"{sdnn:.4f}" if isinstance(sdnn, float) else str(sdnn)
                lorenz_str = f"{lorenz:.4f}" if isinstance(lorenz, float) else str(lorenz)
                f_scaled_str = f"{f_scaled:.4f}" if isinstance(f_scaled, float) else str(f_scaled)
                logger.warning(f"   候補{i+1}: {activity_name}, SDNN={sdnn_str}, Lorenz={lorenz_str}, F_scaled={f_scaled_str}")

            # 元の活動カテゴリを特定
            original_activity_name = activity.get('CatSub', 'unknown')

            # 全ての反実仮想例を評価し、最良の改善案を選択
            best_result = None
            best_improvement = 0

            for idx, cf_row in cf_df.iterrows():
                # 反実仮想例の活動カテゴリを取得（CatSub列から直接取得）
                suggested_activity_name = cf_row.get('CatSub', 'unknown')

                # 活動が変わっていない場合はスキップ
                if suggested_activity_name == original_activity_name:
                    logger.warning(f"   候補{idx}: 活動が元と同じ（{suggested_activity_name}）のでスキップ")
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
                logger.warning(f"✅ DiCE個別処理成功: {best_result['original_activity']} → {best_result['suggested_activity']} (改善: {best_improvement:.2f}点)")
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
                    for i, (idx, cf_row) in enumerate(cf_df.iterrows()):
                        suggested_act = cf_row.get('CatSub', 'unknown')
                        alt_f_scaled = cf_row.get('NASA_F_scaled', 0)
                        alt_f = alt_f_scaled * 20.0
                        imp = current_frustration - alt_f
                        logger.warning(f"    候補{i+1}: {suggested_act}, F値={alt_f:.2f}, 改善={imp:.2f}点")
                return None

        except Exception as e:
            logger.error(f"❌ DiCE個別処理エラー: {e}", exc_info=True)
            logger.warning(f"❌ DiCE個別処理でエラーが発生: {str(e)[:200]}")
            return None

    def generate_hourly_alternatives(self, activities_data: pd.DataFrame,
                                   predictor, target_date: datetime = None, callback=None) -> dict:
        """
        1日の終わりに時間単位でDiCE改善提案を生成

        Args:
            callback: DiCE結果を1件生成するたびに呼び出される関数
        """
        try:
            if target_date is None:
                target_date = datetime.now().date()

            if activities_data.empty:
                logger.warning("時間別DiCE提案: 活動データが空です")
                return self._get_error_hourly_schedule("活動データが空です")

            # 指定日のデータを抽出
            logger.warning(f"🔍 対象日 {target_date} のデータを抽出中...")
            logger.warning(f"🔍 全活動データのタイムスタンプ範囲: {activities_data['Timestamp'].min()} - {activities_data['Timestamp'].max()}")

            day_data = activities_data[
                activities_data['Timestamp'].dt.date == target_date
            ].copy()

            logger.warning(f"🔍 抽出結果: {len(day_data)}件のデータが見つかりました")

            if day_data.empty:
                logger.error(f"❌ 時間別DiCE提案: {target_date}のデータが見つかりません")
                logger.error(f"   activities_data全体の件数: {len(activities_data)}")
                logger.error(f"   Timestampカラムの型: {activities_data['Timestamp'].dtype}")
                return self._get_error_hourly_schedule(f"{target_date}のデータが見つかりません")

            # 時間別の改善提案を生成
            hourly_schedule = []
            total_improvement = 0

            logger.warning(f"🔄 24時間分のDiCE提案を生成開始（対象日に活動がある時間帯のみ処理）")
            activities_processed = 0

            for hour in range(24):
                hour_start = datetime.combine(target_date, datetime.min.time()) + timedelta(hours=hour)
                hour_end = hour_start + timedelta(hours=1)

                # この時間帯の活動データを取得
                hour_activities = day_data[
                    (day_data['Timestamp'] >= hour_start) &
                    (day_data['Timestamp'] < hour_end)
                ]

                if not hour_activities.empty:
                    logger.warning(f"  🔍 {hour}時台: 活動あり、DiCE処理開始...")
                    activities_processed += 1
                    original_activity = hour_activities.iloc[0]
                    idx = activities_data.index[activities_data['Timestamp'] == original_activity['Timestamp']]

                    if len(idx) > 0:
                        # DiCEを使った代替活動の提案
                        import time
                        start_time = time.time()
                        result = self._generate_dice_counterfactual_simple(
                            activities_data, idx[0], original_activity, predictor
                        )
                        elapsed = time.time() - start_time

                        if result:
                            # 【重要】実際のTimestampから時刻を取得（Hourly Logとの一致のため）
                            actual_time = original_activity['Timestamp'].strftime('%H:%M')
                            dice_result = {
                                'hour': hour,
                                'time': actual_time,  # 実際のTimestamp (例: "14:30")
                                'time_range': f"{hour:02d}:00-{hour+1:02d}:00",
                                'original_activity': result['original_activity'],
                                'suggested_activity': result['suggested_activity'],
                                'original_frustration': result['original_frustration'],  # 現在のF値（予測値）
                                'predicted_frustration': result['predicted_frustration'],  # 改善後のF値
                                'improvement': result['improvement'],
                                'confidence': result['confidence']
                            }
                            hourly_schedule.append(dice_result)
                            total_improvement += result['improvement']
                            logger.warning(f"  ✅ {hour}時台: {result['original_activity']} → {result['suggested_activity']} (改善: {result['improvement']:.2f}, 処理時間: {elapsed:.1f}秒)")

                            # コールバック関数が指定されていれば、即座に呼び出す
                            if callback:
                                callback(dice_result)
                        else:
                            logger.warning(f"  ⚠️ {hour}時台: DiCE提案なし（処理時間: {elapsed:.1f}秒）")

            logger.warning(f"🔍 hourly_schedule生成完了: {len(hourly_schedule)}件, total_improvement={total_improvement:.2f}")

            if total_improvement == 0:
                logger.error(f"❌ 時間別DiCE提案: 改善提案を生成できませんでした")
                logger.error(f"   対象日のデータ件数: {len(day_data)}")
                logger.error(f"   hourly_scheduleの長さ: {len(hourly_schedule)}")
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
