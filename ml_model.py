"""
NASA-TLXフラストレーション値予測機械学習モデル
webhooktest.pyの成功パターンに基づくシンプルな実装
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
from typing import Dict, List, Optional

from config import Config

logger = logging.getLogger(__name__)

# document_for_ai.mdに記載されている活動カテゴリリスト
KNOWN_ACTIVITIES = [
    '睡眠', '食事', '身のまわりの用事', '療養・静養', '仕事', '仕事のつきあい',
    '授業・学内の活動', '学校外の学習', '炊事・掃除・洗濯', '買い物', '子どもの世話',
    '家庭雑事', '通勤', '通学', '社会参加', '会話・交際', 'スポーツ', '行楽・散策',
    '趣味・娯楽・教養(インターネット除く)', '趣味・娯楽・教養 のインターネット(動画除く)',
    'インターネット動画', 'テレビ', '録画番組・DVD', 'ラジオ', '新聞',
    '雑誌・漫画・本', '音楽', '休息', 'その他', '不明'
]

class FrustrationPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = []
        self.target_variable = 'NASA_F_scaled'
        self.config = Config()
        self.activity_columns = []  # 活動カテゴリのOne-Hot列名

    def _create_model(self):
        """
        config.MODEL_TYPEに基づいてモデルを作成
        """
        if self.config.MODEL_TYPE == 'Linear':
            logger.info("LinearRegressionモデルを使用します")
            return LinearRegression(n_jobs=-1)
        else:  # デフォルトはRandomForest
            logger.info("RandomForestRegressorモデルを使用します")
            return RandomForestRegressor(
                n_estimators=self.config.N_ESTIMATORS,
                max_depth=self.config.MAX_DEPTH,
                random_state=self.config.RANDOM_STATE,
                n_jobs=-1
            )

    def preprocess_activity_data(self, activity_data: pd.DataFrame) -> pd.DataFrame:
        """
        活動データを前処理 (webhooktest.py形式)
        """
        try:
            if activity_data.empty:
                return pd.DataFrame()

            df = activity_data.copy()
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])

            if 'NASA_F' not in df.columns:
                logger.error("NASA_F列が見つかりません")
                return pd.DataFrame()

            # 数値変換
            df['NASA_F'] = pd.to_numeric(df['NASA_F'], errors='coerce')
            df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')

            # 15分未満の活動を除外
            df = df[df['Duration'] >= 15]

            # 時間特徴量 (webhooktest.py形式)
            df['hour'] = df['Timestamp'].dt.hour
            df['hour_rad'] = 2 * np.pi * df['hour'] / 24
            df['hour_sin'] = np.sin(df['hour_rad'])
            df['hour_cos'] = np.cos(df['hour_rad'])

            # 曜日特徴量
            df['weekday_str'] = df['Timestamp'].dt.strftime('%a')
            df = pd.get_dummies(df, columns=['weekday_str'], prefix='weekday', dtype=int)  # int型に指定

            # NASA_Fのスケーリング (0-20 → 0-1)
            df['NASA_F_scaled'] = df['NASA_F'] / 20.0

            # 活動カテゴリのOne-Hot化 (webhooktest.py形式)
            # CatSubの値を取得し、既知の活動リストでOne-Hot化
            for activity in KNOWN_ACTIVITIES:
                df[f'activity_{activity}'] = (df['CatSub'] == activity).astype(int)

            if self.config.LOG_DATA_OPERATIONS:
                logger.info(f"活動データ前処理完了: {len(df)} 行")

            return df

        except Exception as e:
            logger.error(f"活動データ前処理エラー: {e}")
            return pd.DataFrame()

    def aggregate_fitbit_by_activity(self, activity_data: pd.DataFrame, fitbit_data: pd.DataFrame) -> pd.DataFrame:
        """
        Fitbitデータを活動期間ごとに集計 (webhooktest.py形式のスケーリング)
        """
        try:
            if activity_data.empty:
                return activity_data

            if fitbit_data.empty:
                logger.warning("Fitbitデータが空です。生体情報なしで継続します")
                # Fitbitデータがない場合、SDNN_scaled, Lorenz_Area_scaledをNaNで追加
                activity_data['SDNN_scaled'] = np.nan
                activity_data['Lorenz_Area_scaled'] = np.nan
                return activity_data

            fitbit_data = fitbit_data.copy()
            fitbit_data['Timestamp'] = pd.to_datetime(fitbit_data['Timestamp'])

            # SDNNとLorenz_Areaを数値化
            if 'SDNN' in fitbit_data.columns:
                fitbit_data['SDNN'] = pd.to_numeric(fitbit_data['SDNN'], errors='coerce')
            else:
                fitbit_data['SDNN'] = np.nan

            if 'Lorenz_Area' in fitbit_data.columns:
                fitbit_data['Lorenz_Area'] = pd.to_numeric(fitbit_data['Lorenz_Area'], errors='coerce')
            else:
                fitbit_data['Lorenz_Area'] = np.nan

            # スケーリング (webhooktest.py形式)
            sdnn_max = fitbit_data['SDNN'].max()
            lorenz_max = fitbit_data['Lorenz_Area'].max()

            if sdnn_max > 0:
                fitbit_data['SDNN_scaled'] = fitbit_data['SDNN'] / sdnn_max
            else:
                fitbit_data['SDNN_scaled'] = np.nan

            if lorenz_max > 0:
                fitbit_data['Lorenz_Area_scaled'] = fitbit_data['Lorenz_Area'] / lorenz_max
            else:
                fitbit_data['Lorenz_Area_scaled'] = np.nan

            # 各活動期間のFitbit統計量を計算
            activity_with_fitbit = []

            for idx, activity in activity_data.iterrows():
                start_time = activity['Timestamp']
                duration_minutes = activity['Duration']
                end_time = start_time + timedelta(minutes=duration_minutes)

                # 該当期間のFitbitデータを取得
                fitbit_period = fitbit_data[
                    (fitbit_data['Timestamp'] >= start_time) &
                    (fitbit_data['Timestamp'] <= end_time)
                ]

                # 統計量を計算 (webhooktest.pyはSDNN_scaled, Lorenz_Area_scaledの平均を使用)
                activity_dict = activity.to_dict()
                if not fitbit_period.empty:
                    activity_dict['SDNN_scaled'] = fitbit_period['SDNN_scaled'].mean()
                    activity_dict['Lorenz_Area_scaled'] = fitbit_period['Lorenz_Area_scaled'].mean()
                else:
                    activity_dict['SDNN_scaled'] = np.nan
                    activity_dict['Lorenz_Area_scaled'] = np.nan

                activity_with_fitbit.append(activity_dict)

            result_df = pd.DataFrame(activity_with_fitbit)
            if self.config.LOG_DATA_OPERATIONS:
                logger.info(f"Fitbit統計量化完了: {len(result_df)} 行")
            return result_df

        except Exception as e:
            logger.error(f"Fitbit統計量化エラー: {e}")
            # エラー時もSSDN_scaled, Lorenz_Area_scaledをNaNで追加
            activity_data['SDNN_scaled'] = np.nan
            activity_data['Lorenz_Area_scaled'] = np.nan
            return activity_data

    def check_data_quality(self, df_enhanced: pd.DataFrame) -> Dict:
        """
        学習データの品質をチェック
        """
        try:
            data_quality = {
                'total_samples': len(df_enhanced),
                'is_sufficient': False,
                'quality_level': 'insufficient',
                'warnings': [],
                'recommendations': []
            }

            if len(df_enhanced) < 10:
                data_quality['warnings'].append(f"データ数が不足しています（{len(df_enhanced)}件/最低10件必要）")
                data_quality['recommendations'].append("より多くの活動データを記録してください。")
            elif len(df_enhanced) < 30:
                data_quality['quality_level'] = 'minimal'
                data_quality['is_sufficient'] = True
                data_quality['warnings'].append(f"データ数が少ないため、予測精度が低い可能性があります（{len(df_enhanced)}件）")
                data_quality['recommendations'].append("30件以上のデータで精度が向上します。")
            elif len(df_enhanced) < 100:
                data_quality['quality_level'] = 'moderate'
                data_quality['is_sufficient'] = True
            else:
                data_quality['quality_level'] = 'good'
                data_quality['is_sufficient'] = True

            # フラストレーション値の分散チェック
            if 'NASA_F' in df_enhanced.columns:
                frustration_std = df_enhanced['NASA_F'].std()
                if frustration_std < 1.0:
                    data_quality['warnings'].append(f"フラストレーション値のバラつきが小さいです（標準偏差: {frustration_std:.2f}）")
                    data_quality['recommendations'].append("様々な状況での活動データを記録してください。")

            return data_quality

        except Exception as e:
            logger.error(f"データ品質チェックエラー: {e}")
            return {
                'total_samples': 0,
                'is_sufficient': False,
                'quality_level': 'error',
                'warnings': [str(e)],
                'recommendations': []
            }

    def train_model(self, df_enhanced: pd.DataFrame) -> Dict:
        """
        モデルを訓練 (webhooktest.py形式のシンプルな実装)
        """
        try:
            data_quality = self.check_data_quality(df_enhanced)

            if len(df_enhanced) < 10:
                raise ValueError(f"訓練には最低10個のデータが必要です（現在: {len(df_enhanced)}件）")

            # NaN値を含む行を除外
            required_cols = ['SDNN_scaled', 'Lorenz_Area_scaled', 'NASA_F_scaled']
            df_clean = df_enhanced.dropna(subset=required_cols)

            if df_clean.empty:
                raise ValueError("有効なデータがありません。SDNN, Lorenz_Area, NASA_Fがすべて必要です。")

            # 活動カテゴリ列を取得
            activity_cols = [col for col in df_clean.columns if col.startswith('activity_')]
            self.activity_columns = activity_cols

            # 曜日列を取得
            weekday_cols = [col for col in df_clean.columns if col.startswith('weekday_')]

            # 時間特徴量
            time_features = ['hour_sin', 'hour_cos']

            # 特徴量: webhooktest.py形式
            feature_list = ['SDNN_scaled', 'Lorenz_Area_scaled'] + activity_cols + time_features + weekday_cols
            self.feature_columns = feature_list

            # 特徴量とターゲットを抽出
            X = df_clean[self.feature_columns]
            y = df_clean['NASA_F_scaled']

            # 訓練/テストデータに分割
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.config.RANDOM_STATE)

            # モデルを作成して訓練
            self.model = self._create_model()
            self.model.fit(X_train, y_train)

            # 評価
            train_pred = self.model.predict(X_train)
            test_pred = self.model.predict(X_test)

            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)

            results = {
                'train_rmse': float(train_rmse),
                'test_rmse': float(test_rmse),
                'train_r2': float(train_r2),
                'test_r2': float(test_r2),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_count': len(self.feature_columns),
                'data_quality': data_quality,
                'model_type': self.config.MODEL_TYPE
            }

            # feature_importanceはRandomForestのみ
            if hasattr(self.model, 'feature_importances_'):
                results['feature_importance'] = dict(zip(self.feature_columns, self.model.feature_importances_))
            elif hasattr(self.model, 'coef_'):
                # LinearRegressionの場合は係数を記録
                results['feature_coefficients'] = dict(zip(self.feature_columns, self.model.coef_))

            if self.config.LOG_MODEL_TRAINING:
                logger.info(f"モデル訓練完了 - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}, Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}")

            return results

        except Exception as e:
            logger.error(f"モデル訓練エラー: {e}")
            return {'error': str(e)}

    def walk_forward_validation_train(self, df_enhanced: pd.DataFrame) -> Dict:
        """
        Walk Forward Validationによる訓練 (webhooktest.py形式の特徴量)
        過去のデータで訓練し、現在のデータで評価
        """
        try:
            data_quality = self.check_data_quality(df_enhanced)

            if len(df_enhanced) < 10:
                raise ValueError(f"Walk Forward Validationには最低10個のデータが必要です（現在: {len(df_enhanced)}件）")

            # NaN値を含む行を除外
            required_cols = ['SDNN_scaled', 'Lorenz_Area_scaled', 'NASA_F_scaled']
            df_clean = df_enhanced.dropna(subset=required_cols).copy()

            if df_clean.empty:
                raise ValueError("有効なデータがありません。SDNN, Lorenz_Area, NASA_Fがすべて必要です。")

            # 活動カテゴリ列を取得
            activity_cols = [col for col in df_clean.columns if col.startswith('activity_')]
            self.activity_columns = activity_cols

            # 曜日列を取得
            weekday_cols = [col for col in df_clean.columns if col.startswith('weekday_')]

            # 時間特徴量
            time_features = ['hour_sin', 'hour_cos']

            # 特徴量リスト (webhooktest.py形式)
            feature_list = ['SDNN_scaled', 'Lorenz_Area_scaled'] + activity_cols + time_features + weekday_cols
            self.feature_columns = feature_list

            # Walk Forward Validation: 過去のデータで訓練、現在を予測
            predictions = []
            actuals = []

            # 最初の30%はウォームアップ期間として使用
            start_idx = max(10, int(len(df_clean) * 0.3))

            for i in range(start_idx, len(df_clean)):
                # 過去のデータ(i以前)で訓練
                train_data = df_clean.iloc[:i]
                X_train = train_data[self.feature_columns]
                y_train = train_data['NASA_F_scaled']

                # モデル訓練
                model = self._create_model()
                model.fit(X_train, y_train)

                # 現在(i番目)のデータで予測
                current_data = df_clean.iloc[i:i+1]
                X_current = current_data[self.feature_columns]
                y_current = current_data['NASA_F_scaled'].values[0]

                prediction = model.predict(X_current)[0]

                predictions.append(prediction)
                actuals.append(y_current)

            if len(predictions) == 0:
                raise ValueError("有効な予測が生成されませんでした")

            # 最終モデル: 全データで訓練
            X_all = df_clean[self.feature_columns]
            y_all = df_clean['NASA_F_scaled']

            self.model = self._create_model()
            self.model.fit(X_all, y_all)

            # 評価メトリクス
            predictions_array = np.array(predictions)
            actuals_array = np.array(actuals)

            rmse = np.sqrt(mean_squared_error(actuals_array, predictions_array))
            mae = mean_absolute_error(actuals_array, predictions_array)
            r2 = r2_score(actuals_array, predictions_array)

            # 予測値の多様性チェック
            prediction_std = np.std(predictions_array)
            prediction_unique = len(np.unique(np.round(predictions_array, 2)))

            results = {
                'walk_forward_rmse': float(rmse),
                'walk_forward_mae': float(mae),
                'walk_forward_r2': float(r2),
                'total_predictions': len(predictions),
                'training_samples': len(df_clean),
                'feature_count': len(self.feature_columns),
                'data_quality': data_quality,
                'model_type': self.config.MODEL_TYPE,
                'prediction_diversity': {
                    'std': float(prediction_std),
                    'unique_values': int(prediction_unique),
                    'is_diverse': prediction_std > 0.05 and prediction_unique > 3
                }
            }

            # feature_importanceはRandomForestのみ
            if hasattr(self.model, 'feature_importances_'):
                results['feature_importance'] = dict(zip(self.feature_columns, self.model.feature_importances_))
            elif hasattr(self.model, 'coef_'):
                # LinearRegressionの場合は係数を記録
                results['feature_coefficients'] = dict(zip(self.feature_columns, self.model.coef_))

            if self.config.LOG_MODEL_TRAINING:
                logger.info(f"Walk Forward Validation完了 - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.3f}, 予測数: {len(predictions)}")

            return results

        except Exception as e:
            logger.error(f"Walk Forward Validation訓練エラー: {e}")
            return {'error': str(e)}

    def predict_single_activity(self, activity_category: str, duration: int = 60, current_time: datetime = None) -> dict:
        """
        単一の活動に対してフラストレーション値を予測 (webhooktest.py形式)

        注意: 履歴データがないため、SDNN_scaled, Lorenz_Area_scaledに固定値を使用します。
        predict_with_historyの使用を推奨します。
        """
        try:
            if self.model is None:
                return {
                    'predicted_frustration': np.nan,
                    'confidence': 0.0,
                    'error': 'モデルが訓練されていません'
                }

            if current_time is None:
                current_time = datetime.now()

            # 特徴量を構築
            features = {}

            # 時間特徴量
            hour_rad = 2 * np.pi * current_time.hour / 24
            features['hour_sin'] = np.sin(hour_rad)
            features['hour_cos'] = np.cos(hour_rad)

            # 曜日のOne-Hot (現在の曜日のみ1、他は0)
            weekday_str = current_time.strftime('%a')
            for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:
                features[f'weekday_{day}'] = 1 if weekday_str == day else 0

            # 活動カテゴリのOne-Hot (指定された活動のみ1、他は0)
            for activity in KNOWN_ACTIVITIES:
                features[f'activity_{activity}'] = 1 if activity_category == activity else 0

            # 生体情報 (履歴データがないため、訓練データの平均値を使用)
            # これは推奨されない方法です。predict_with_historyを使用してください。
            features['SDNN_scaled'] = 0.5
            features['Lorenz_Area_scaled'] = 0.5
            logger.warning(f"生体情報に固定値を使用しています。predict_with_historyの使用を推奨します。")

            # DataFrameに変換し、モデルの特徴量順に並べる
            feature_df = pd.DataFrame([features])
            for col in self.feature_columns:
                if col not in feature_df.columns:
                    feature_df[col] = 0.0
            feature_df = feature_df[self.feature_columns]

            # 予測実行 (0-1スケール)
            prediction_scaled = self.model.predict(feature_df)[0]

            # 元のスケール (0-20) に戻す
            prediction = prediction_scaled * 20.0

            return {
                'predicted_frustration': float(prediction),
                'confidence': 0.3,  # 固定値使用のため低信頼度
                'activity_category': activity_category,
                'duration': duration,
                'timestamp': current_time,
                'used_historical_data': False
            }

        except Exception as e:
            logger.error(f"単一活動予測エラー: {e}")
            return {
                'predicted_frustration': np.nan,
                'confidence': 0.0,
                'error': str(e)
            }

    def predict_with_history(self, activity_category: str, duration: int, current_time: datetime, historical_data: pd.DataFrame) -> dict:
        """
        過去の履歴データを使用して予測 (webhooktest.py形式)

        Args:
            activity_category: 活動カテゴリ
            duration: 活動時間（分）
            current_time: 現在時刻
            historical_data: 過去のデータ (aggregate_fitbit_by_activityの出力)

        Returns:
            予測結果
        """
        try:
            if self.model is None:
                return self.predict_single_activity(activity_category, duration, current_time)

            if historical_data.empty:
                logger.warning("履歴データが空のため、固定値を使用します")
                return self.predict_single_activity(activity_category, duration, current_time)

            # 特徴量を構築
            features = {}

            # 時間特徴量
            hour_rad = 2 * np.pi * current_time.hour / 24
            features['hour_sin'] = np.sin(hour_rad)
            features['hour_cos'] = np.cos(hour_rad)

            # 曜日のOne-Hot
            weekday_str = current_time.strftime('%a')
            for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:
                features[f'weekday_{day}'] = 1 if weekday_str == day else 0

            # 活動カテゴリのOne-Hot
            for activity in KNOWN_ACTIVITIES:
                features[f'activity_{activity}'] = 1 if activity_category == activity else 0

            # 生体情報: 履歴データの平均値を使用
            if 'SDNN_scaled' in historical_data.columns:
                sdnn_mean = historical_data['SDNN_scaled'].dropna().mean()
                if pd.isna(sdnn_mean):
                    logger.error("SDNN_scaledの有効なデータがありません")
                    return {
                        'predicted_frustration': np.nan,
                        'confidence': 0.0,
                        'error': 'SDNN_scaledの有効なデータがありません',
                        'activity_category': activity_category,
                        'duration': duration,
                        'timestamp': current_time,
                        'used_historical_data': False
                    }
                features['SDNN_scaled'] = sdnn_mean
            else:
                logger.error("SDNN_scaled列が見つかりません")
                return {
                    'predicted_frustration': np.nan,
                    'confidence': 0.0,
                    'error': 'SDNN_scaled列が見つかりません',
                    'activity_category': activity_category,
                    'duration': duration,
                    'timestamp': current_time,
                    'used_historical_data': False
                }

            if 'Lorenz_Area_scaled' in historical_data.columns:
                lorenz_mean = historical_data['Lorenz_Area_scaled'].dropna().mean()
                if pd.isna(lorenz_mean):
                    logger.error("Lorenz_Area_scaledの有効なデータがありません")
                    return {
                        'predicted_frustration': np.nan,
                        'confidence': 0.0,
                        'error': 'Lorenz_Area_scaledの有効なデータがありません',
                        'activity_category': activity_category,
                        'duration': duration,
                        'timestamp': current_time,
                        'used_historical_data': False
                    }
                features['Lorenz_Area_scaled'] = lorenz_mean
            else:
                logger.error("Lorenz_Area_scaled列が見つかりません")
                return {
                    'predicted_frustration': np.nan,
                    'confidence': 0.0,
                    'error': 'Lorenz_Area_scaled列が見つかりません',
                    'activity_category': activity_category,
                    'duration': duration,
                    'timestamp': current_time,
                    'used_historical_data': False
                }

            # DataFrameに変換
            feature_df = pd.DataFrame([features])
            for col in self.feature_columns:
                if col not in feature_df.columns:
                    feature_df[col] = 0
            feature_df = feature_df[self.feature_columns]

            # 予測実行 (0-1スケール)
            prediction_scaled = self.model.predict(feature_df)[0]

            # 元のスケール (0-20) に戻す
            prediction = prediction_scaled * 20.0

            # NaN/Infのバリデーション
            if np.isnan(prediction) or np.isinf(prediction):
                logger.error(f"予測値が不正です (NaN/Inf): {prediction}")
                return {
                    'predicted_frustration': np.nan,
                    'confidence': 0.0,
                    'error': '予測値の計算に失敗しました',
                    'activity_category': activity_category,
                    'duration': duration,
                    'timestamp': current_time,
                    'used_historical_data': True
                }

            return {
                'predicted_frustration': float(prediction),
                'confidence': 0.7,
                'activity_category': activity_category,
                'duration': duration,
                'timestamp': current_time,
                'used_historical_data': True,
                'historical_records': len(historical_data)
            }

        except Exception as e:
            logger.error(f"履歴データを使った予測エラー: {e}")
            return self.predict_single_activity(activity_category, duration, current_time)

    def get_prediction_confidence(self, prediction: float, features: dict) -> float:
        """
        予測の信頼度を計算
        """
        try:
            # 基本信頼度
            base_confidence = 0.7 if 1 <= prediction <= 20 else 0.3

            # 特徴量の完全性
            feature_completeness = len([v for v in features.values() if v != 0]) / len(features)
            completeness_bonus = feature_completeness * 0.2

            confidence = min(0.95, base_confidence + completeness_bonus)
            return confidence

        except Exception as e:
            logger.error(f"信頼度計算エラー: {e}")
            return 0.0

    def save_model(self, filepath: str):
        """モデルを保存"""
        try:
            model_data = {
                'model': self.model,
                'feature_columns': self.feature_columns,
                'activity_columns': self.activity_columns
            }
            joblib.dump(model_data, filepath)
            if self.config.LOG_MODEL_TRAINING:
                logger.info(f"モデルを保存しました: {filepath}")
        except Exception as e:
            logger.error(f"モデル保存エラー: {e}")

    def load_model(self, filepath: str) -> bool:
        """モデルを読み込み"""
        try:
            if os.path.exists(filepath):
                model_data = joblib.load(filepath)
                self.model = model_data['model']
                self.feature_columns = model_data['feature_columns']
                self.activity_columns = model_data.get('activity_columns', [])
                if self.config.LOG_MODEL_TRAINING:
                    logger.info(f"モデルを読み込みました: {filepath}")
                return True
            else:
                logger.warning(f"モデルファイルが見つかりません: {filepath}")
                return False
        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
            return False
