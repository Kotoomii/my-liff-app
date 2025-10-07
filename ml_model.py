"""
NASA-TLXフラストレーション値予測機械学習モデル
Walk Forward Validation対応、行動変更タイミングでの予測
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from typing import Dict, List, Tuple, Optional

from config import Config

logger = logging.getLogger(__name__)

class FrustrationPredictor:
    def __init__(self):
        self.model = None
        self.encoders = {}
        self.feature_columns = []
        self.target_variable = 'NASA_F'  # フラストレーション値のみを予測
        self.config = Config()
        self.walk_forward_history = []  # Walk Forward Validation用の履歴
        
    def preprocess_activity_data(self, activity_data: pd.DataFrame) -> pd.DataFrame:
        """
        活動データを行動単位に前処理
        最小区切り時間は15分
        """
        try:
            if activity_data.empty:
                return pd.DataFrame()
                
            # データ型変換
            activity_data = activity_data.copy()
            activity_data['Timestamp'] = pd.to_datetime(activity_data['Timestamp'])
            
            # フラストレーション値のチェック
            if 'NASA_F' not in activity_data.columns:
                logger.error("NASA_F列が見つかりません")
                return pd.DataFrame()
            
            # 数値変換
            activity_data['NASA_F'] = pd.to_numeric(activity_data['NASA_F'], errors='coerce')
            activity_data['Duration'] = pd.to_numeric(activity_data['Duration'], errors='coerce')
            
            # 15分未満の活動を除外
            activity_data = activity_data[activity_data['Duration'] >= 15]
            
            # 行動変更タイミングの検出
            activity_data = activity_data.sort_values('Timestamp')
            activity_data['activity_change'] = (
                activity_data['CatSub'] != activity_data['CatSub'].shift(1)
            ).astype(int)
            
            # 時間特徴量の追加
            activity_data['hour'] = activity_data['Timestamp'].dt.hour
            activity_data['dayofweek'] = activity_data['Timestamp'].dt.dayofweek
            activity_data['is_weekend'] = (activity_data['dayofweek'] >= 5).astype(int)
            
            return activity_data
            
        except Exception as e:
            logger.error(f"活動データ前処理エラー: {e}")
            return pd.DataFrame()
    
    def aggregate_fitbit_by_activity(self, activity_data: pd.DataFrame, fitbit_data: pd.DataFrame) -> pd.DataFrame:
        """
        Fitbitデータを行動の長さごとに統計量化
        """
        try:
            if activity_data.empty or fitbit_data.empty:
                return activity_data
                
            fitbit_data = fitbit_data.copy()
            fitbit_data['Timestamp'] = pd.to_datetime(fitbit_data['Timestamp'])
            fitbit_data['Lorenz_Area'] = pd.to_numeric(fitbit_data['Lorenz_Area'], errors='coerce')
            
            # 各活動期間のFitbitデータ統計量を計算
            activity_with_fitbit = []
            
            for idx, activity in activity_data.iterrows():
                start_time = pd.to_datetime(activity['Timestamp'])
                duration_minutes = activity['Duration']
                end_time = start_time + timedelta(minutes=duration_minutes)
                
                # 該当期間のFitbitデータを取得
                fitbit_period = fitbit_data[
                    (fitbit_data['Timestamp'] >= start_time) & 
                    (fitbit_data['Timestamp'] <= end_time)
                ]
                
                # Fitbit統計量を計算
                if not fitbit_period.empty:
                    lorenz_stats = {
                        'lorenz_mean': fitbit_period['Lorenz_Area'].mean(),
                        'lorenz_std': fitbit_period['Lorenz_Area'].std(),
                        'lorenz_min': fitbit_period['Lorenz_Area'].min(),
                        'lorenz_max': fitbit_period['Lorenz_Area'].max(),
                        'lorenz_median': fitbit_period['Lorenz_Area'].median(),
                        'lorenz_q25': fitbit_period['Lorenz_Area'].quantile(0.25),
                        'lorenz_q75': fitbit_period['Lorenz_Area'].quantile(0.75),
                        'lorenz_count': len(fitbit_period)
                    }
                else:
                    # データがない場合はデフォルト値
                    lorenz_stats = {
                        'lorenz_mean': 8000.0,
                        'lorenz_std': 0.0,
                        'lorenz_min': 8000.0,
                        'lorenz_max': 8000.0,
                        'lorenz_median': 8000.0,
                        'lorenz_q25': 8000.0,
                        'lorenz_q75': 8000.0,
                        'lorenz_count': 0
                    }
                
                # 活動データに統計量を追加
                activity_enhanced = activity.to_dict()
                activity_enhanced.update(lorenz_stats)
                activity_with_fitbit.append(activity_enhanced)
            
            result_df = pd.DataFrame(activity_with_fitbit)
            if self.config.LOG_DATA_OPERATIONS:
                logger.info(f"Fitbit統計量化完了: {len(result_df)} 行")
            return result_df
            
        except Exception as e:
            logger.error(f"Fitbit統計量化エラー: {e}")
            return activity_data
    
    def create_features_for_activity(self, df_with_fitbit: pd.DataFrame, current_idx: int, 
                                   lookback_hours: int = 24) -> Optional[Dict]:
        """
        指定した行動変更タイミングの特徴量を作成
        過去24時間の活動データから特徴量を生成
        """
        try:
            if current_idx >= len(df_with_fitbit):
                return None
                
            current_activity = df_with_fitbit.iloc[current_idx]
            current_time = current_activity['Timestamp']
            lookback_time = current_time - timedelta(hours=lookback_hours)
            
            # 過去24時間のデータを取得
            historical_data = df_with_fitbit[
                (df_with_fitbit['Timestamp'] >= lookback_time) & 
                (df_with_fitbit['Timestamp'] < current_time)
            ]
            
            if historical_data.empty:
                return None
            
            # 特徴量を構築
            features = {}
            
            # 現在の活動の基本特徴量
            features['current_hour'] = current_activity['hour']
            features['current_dayofweek'] = current_activity['dayofweek']
            features['current_is_weekend'] = current_activity['is_weekend']
            features['current_duration'] = current_activity['Duration']
            
            # カテゴリ特徴量のエンコード
            if 'CatSub' in current_activity and pd.notna(current_activity['CatSub']):
                activity_name = str(current_activity['CatSub'])
                if 'current_activity' not in self.encoders:
                    self.encoders['current_activity'] = LabelEncoder()
                    # 既知の活動カテゴリで初期化
                    known_activities = ['睡眠', '食事', '仕事', '休憩', '運動', '通勤', '娯楽', 'その他']
                    self.encoders['current_activity'].fit(known_activities)
                
                try:
                    features['current_activity'] = self.encoders['current_activity'].transform([activity_name])[0]
                except ValueError:
                    # 未知のカテゴリは'その他'として扱う
                    features['current_activity'] = self.encoders['current_activity'].transform(['その他'])[0]
            else:
                features['current_activity'] = self.encoders['current_activity'].transform(['その他'])[0] if 'current_activity' in self.encoders else 7
            
            # 過去24時間の統計特徴量
            features['hist_avg_frustration'] = historical_data['NASA_F'].mean()
            features['hist_std_frustration'] = historical_data['NASA_F'].std()
            features['hist_max_frustration'] = historical_data['NASA_F'].max()
            features['hist_min_frustration'] = historical_data['NASA_F'].min()
            features['hist_total_duration'] = historical_data['Duration'].sum()
            features['hist_avg_duration'] = historical_data['Duration'].mean()
            features['hist_activity_changes'] = historical_data['activity_change'].sum()
            
            # Fitbit統計特徴量（過去24時間の平均）
            fitbit_cols = ['lorenz_mean', 'lorenz_std', 'lorenz_min', 'lorenz_max', 
                          'lorenz_median', 'lorenz_q25', 'lorenz_q75']
            for col in fitbit_cols:
                if col in historical_data.columns:
                    features[f'hist_{col}'] = historical_data[col].mean()
                else:
                    features[f'hist_{col}'] = 8000.0
            
            # 時間帯別特徴量（過去24時間）
            for start_hour, end_hour, label in [(0, 6, 'night'), (6, 12, 'morning'), 
                                              (12, 18, 'afternoon'), (18, 24, 'evening')]:
                hour_data = historical_data[
                    (historical_data['hour'] >= start_hour) & 
                    (historical_data['hour'] < end_hour)
                ]
                features[f'hist_{label}_avg_frustration'] = hour_data['NASA_F'].mean() if not hour_data.empty else 10.0
                features[f'hist_{label}_duration'] = hour_data['Duration'].sum()
            
            # NaN値を適切なデフォルト値で置換
            for key, value in features.items():
                if pd.isna(value):
                    if 'frustration' in key:
                        features[key] = 10.0
                    elif 'duration' in key:
                        features[key] = 0.0
                    elif 'lorenz' in key:
                        features[key] = 8000.0
                    else:
                        features[key] = 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"特徴量作成エラー: {e}")
            return None
    
    def check_data_quality(self, df_enhanced: pd.DataFrame) -> Dict:
        """
        学習データの品質と十分性をチェック
        """
        try:
            data_quality = {
                'total_samples': len(df_enhanced),
                'is_sufficient': False,
                'quality_level': 'insufficient',
                'warnings': [],
                'recommendations': []
            }

            # データ数の評価
            if len(df_enhanced) < 10:
                data_quality['warnings'].append(f"データ数が不足しています（{len(df_enhanced)}件/最低10件必要）")
                data_quality['recommendations'].append("より多くの活動データを記録してください。最低10件以上が必要です。")
            elif len(df_enhanced) < 30:
                data_quality['quality_level'] = 'minimal'
                data_quality['is_sufficient'] = True
                data_quality['warnings'].append(f"データ数が少ないため、予測精度が低い可能性があります（{len(df_enhanced)}件）")
                data_quality['recommendations'].append("30件以上のデータで精度が向上します。")
            elif len(df_enhanced) < 100:
                data_quality['quality_level'] = 'moderate'
                data_quality['is_sufficient'] = True
                data_quality['warnings'].append("データ数は適切ですが、さらに増やすとより正確な予測が可能です。")
            else:
                data_quality['quality_level'] = 'good'
                data_quality['is_sufficient'] = True

            # フラストレーション値の分散チェック
            if 'NASA_F' in df_enhanced.columns:
                frustration_std = df_enhanced['NASA_F'].std()
                frustration_unique = df_enhanced['NASA_F'].nunique()

                if frustration_std < 1.0:
                    data_quality['warnings'].append(f"フラストレーション値のバラつきが小さく、モデルが同じ値を予測する可能性があります（標準偏差: {frustration_std:.2f}）")
                    data_quality['recommendations'].append("様々な状況での活動データを記録してください。")

                if frustration_unique < 3:
                    data_quality['warnings'].append(f"フラストレーション値の種類が少なすぎます（{frustration_unique}種類）")
                    data_quality['recommendations'].append("異なるフラストレーション値を持つデータを追加してください。")

            # 活動の多様性チェック
            if 'CatSub' in df_enhanced.columns:
                activity_unique = df_enhanced['CatSub'].nunique()
                if activity_unique < 3:
                    data_quality['warnings'].append(f"活動の種類が少なすぎます（{activity_unique}種類）")
                    data_quality['recommendations'].append("様々な種類の活動データを記録してください。")

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

    def walk_forward_validation_train(self, df_enhanced: pd.DataFrame) -> Dict:
        """
        Walk Forward Validationによる学習
        各行動変更タイミングで、過去24時間のデータで学習し、現在のフラストレーション値を予測
        """
        try:
            # データ品質チェック
            data_quality = self.check_data_quality(df_enhanced)

            if len(df_enhanced) < 10:
                raise ValueError(f"Walk Forward Validationには最低10個のデータポイントが必要です（現在: {len(df_enhanced)}件）")
            
            predictions = []
            actuals = []
            training_results = []
            
            # 最初の数個のデータポイントはスキップ（履歴不足のため）
            start_idx = max(5, len(df_enhanced) // 4)  # データの1/4は履歴として使用
            
            for i in range(start_idx, len(df_enhanced)):
                current_activity = df_enhanced.iloc[i]
                
                # 行動変更タイミングでない場合はスキップ
                if current_activity.get('activity_change', 0) == 0:
                    continue
                
                # 訓練用の特徴量とターゲットを準備
                train_features = []
                train_targets = []
                
                # 過去のデータポイントから訓練データを作成
                for j in range(max(0, i - 50), i):  # 最大過去50個の行動から学習
                    features = self.create_features_for_activity(df_enhanced, j)
                    if features is not None:
                        train_features.append(features)
                        train_targets.append(df_enhanced.iloc[j]['NASA_F'])
                
                if len(train_features) < 5:  # 最低5個の訓練データが必要
                    continue
                
                # 特徴量データフレームを作成
                train_df = pd.DataFrame(train_features)
                train_targets = np.array(train_targets)
                
                # 特徴量列を記録
                if not self.feature_columns:
                    self.feature_columns = list(train_df.columns)
                
                # 不足している特徴量列を0で埋める
                for col in self.feature_columns:
                    if col not in train_df.columns:
                        train_df[col] = 0.0
                
                train_df = train_df[self.feature_columns]
                
                # モデル学習
                model = RandomForestRegressor(
                    n_estimators=50,  # 高速化のため少なめ
                    max_depth=10,
                    random_state=self.config.RANDOM_STATE,
                    n_jobs=1
                )
                model.fit(train_df, train_targets)
                
                # 現在の行動について予測
                current_features = self.create_features_for_activity(df_enhanced, i)
                if current_features is not None:
                    pred_df = pd.DataFrame([current_features])
                    
                    # 不足している特徴量列を0で埋める
                    for col in self.feature_columns:
                        if col not in pred_df.columns:
                            pred_df[col] = 0.0
                    
                    pred_df = pred_df[self.feature_columns]
                    prediction = model.predict(pred_df)[0]
                    actual = current_activity['NASA_F']
                    
                    predictions.append(prediction)
                    actuals.append(actual)
                    
                    # 結果を記録
                    training_results.append({
                        'timestamp': current_activity['Timestamp'],
                        'actual': actual,
                        'predicted': prediction,
                        'activity': current_activity.get('CatSub', 'unknown'),
                        'training_size': len(train_features)
                    })
            
            if len(predictions) == 0:
                raise ValueError("有効な予測が生成されませんでした")
            
            # 最後のモデルを保存
            self.model = model
            
            # 評価メトリクス計算
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
            r2 = r2_score(actuals, predictions)
            
            self.walk_forward_history = training_results
            
            # 予測値の多様性チェック
            prediction_std = np.std(predictions)
            prediction_unique = len(np.unique(np.round(predictions, 1)))

            results = {
                'walk_forward_rmse': rmse,
                'walk_forward_mae': mae,
                'walk_forward_r2': r2,
                'total_predictions': len(predictions),
                'feature_importance': dict(zip(self.feature_columns, model.feature_importances_)),
                'prediction_history': training_results[-10:],  # 最新10件のみ返す
                'data_quality': data_quality,
                'prediction_diversity': {
                    'std': float(prediction_std),
                    'unique_values': int(prediction_unique),
                    'is_diverse': prediction_std > 1.0 and prediction_unique > 3
                }
            }

            # 予測値が単調な場合の警告
            if prediction_std < 1.0:
                logger.warning(f"⚠️ 予測値のバラつきが小さいです（標準偏差: {prediction_std:.2f}）。データ不足またはデータの偏りが原因の可能性があります。")

            if prediction_unique < 3:
                logger.warning(f"⚠️ 予測値の種類が少なすぎます（{prediction_unique}種類）。モデルが同じ値を繰り返し予測しています。")

            if self.config.LOG_MODEL_TRAINING:
                logger.info(f"Walk Forward Validation完了 - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}, データ品質: {data_quality['quality_level']}")
            return results
            
        except Exception as e:
            logger.error(f"Walk Forward Validation学習エラー: {e}")
            return {}
    
    def predict_frustration_at_activity_change(self, df_enhanced: pd.DataFrame, 
                                             target_timestamp: datetime = None) -> Dict:
        """
        指定した時刻（行動変更タイミング）でのフラストレーション値を予測
        """
        try:
            if self.model is None:
                raise ValueError("モデルが学習されていません")
            
            if target_timestamp is None:
                target_timestamp = datetime.now()
            
            # 最も近い行動変更タイミングを見つける
            df_changes = df_enhanced[df_enhanced['activity_change'] == 1].copy()
            
            if df_changes.empty:
                raise ValueError("行動変更タイミングが見つかりません")
            
            # 指定時刻に最も近い行動変更を選択
            df_changes['time_diff'] = abs((df_changes['Timestamp'] - target_timestamp).dt.total_seconds())
            closest_idx = df_changes['time_diff'].idxmin()
            
            # 特徴量を作成
            original_idx = df_enhanced.index.get_loc(closest_idx)
            features = self.create_features_for_activity(df_enhanced, original_idx)
            
            if features is None:
                raise ValueError("特徴量の作成に失敗しました")
            
            # 予測実行
            pred_df = pd.DataFrame([features])
            
            # 不足している特徴量列を0で埋める
            for col in self.feature_columns:
                if col not in pred_df.columns:
                    pred_df[col] = 0.0
            
            pred_df = pred_df[self.feature_columns]
            prediction = self.model.predict(pred_df)[0]
            
            # 結果を返す
            target_activity = df_enhanced.loc[closest_idx]
            
            return {
                'predicted_frustration': float(prediction),
                'actual_frustration': float(target_activity['NASA_F']) if pd.notna(target_activity['NASA_F']) else None,
                'activity': target_activity.get('CatSub', 'unknown'),
                'timestamp': target_activity['Timestamp'],
                'duration': target_activity.get('Duration', 0),
                'confidence': min(0.9, 0.5 + 0.4 * (1 - abs(prediction - 50) / 50)),  # 信頼度の推定
                'features_used': len(self.feature_columns)
            }
            
        except Exception as e:
            logger.error(f"フラストレーション値予測エラー: {e}")
            return {
                'predicted_frustration': 50.0,
                'actual_frustration': None,
                'activity': 'unknown',
                'timestamp': target_timestamp,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def get_activity_change_timestamps(self, df_enhanced: pd.DataFrame, 
                                     hours_back: int = 24) -> List[datetime]:
        """
        過去指定時間内の行動変更タイミングを取得
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            recent_data = df_enhanced[df_enhanced['Timestamp'] >= cutoff_time]
            
            change_points = recent_data[recent_data['activity_change'] == 1]['Timestamp'].tolist()
            return sorted(change_points)
            
        except Exception as e:
            logger.error(f"行動変更タイミング取得エラー: {e}")
            return []
    
    def save_model(self, filepath: str):
        """モデルと関連データを保存"""
        try:
            model_data = {
                'model': self.model,
                'encoders': self.encoders,
                'feature_columns': self.feature_columns,
                'walk_forward_history': self.walk_forward_history
            }
            joblib.dump(model_data, filepath)
            if self.config.LOG_MODEL_TRAINING:
                logger.info(f"モデルを保存しました: {filepath}")
        except Exception as e:
            logger.error(f"モデル保存エラー: {e}")
    
    def load_model(self, filepath: str) -> bool:
        """モデルと関連データを読み込み"""
        try:
            if os.path.exists(filepath):
                model_data = joblib.load(filepath)
                self.model = model_data['model']
                self.encoders = model_data['encoders']
                self.feature_columns = model_data['feature_columns']
                self.walk_forward_history = model_data.get('walk_forward_history', [])
                if self.config.LOG_MODEL_TRAINING:
                    logger.info(f"モデルを読み込みました: {filepath}")
                return True
            else:
                logger.warning(f"モデルファイルが見つかりません: {filepath}")
                return False
        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
            return False
    
    def create_features_for_new_activity(self, activity_category: str, duration: int = 60,
                                       current_time: datetime = None) -> dict:
        """
        新しい活動に対して特徴量を作成（リアルタイム予測用）

        ⚠️ WARNING: この関数は訓練時と異なる特徴量を作成します
        訓練時は過去24時間の統計を使用しますが、予測時は使用できません
        そのため、予測値が偏る可能性があります
        """
        if current_time is None:
            current_time = datetime.now()

        # 基本的な時間特徴量
        features = {
            'current_duration': duration,
            'current_hour': current_time.hour,
            'current_dayofweek': current_time.weekday(),
            'current_is_weekend': 1 if current_time.weekday() >= 5 else 0
        }

        # 活動カテゴリーのエンコーディング
        if hasattr(self, 'encoders') and 'current_activity' in self.encoders:
            try:
                # 既知の活動カテゴリで変換を試みる
                encoded_cat = self.encoders['current_activity'].transform([activity_category])[0]
                features['current_activity'] = encoded_cat
            except ValueError:
                # 未知のカテゴリは'その他'として扱う
                try:
                    features['current_activity'] = self.encoders['current_activity'].transform(['その他'])[0]
                except:
                    features['current_activity'] = 0
        else:
            features['current_activity'] = 0

        # 過去24時間の統計特徴量（デフォルト値）
        # 注意: 実際の履歴データがないため、平均的な値を使用
        # これが予測値が偏る主な原因です
        features['hist_avg_frustration'] = 10.0  # NASA_Fの典型的な平均値
        features['hist_std_frustration'] = 2.0   # 標準偏差
        features['hist_max_frustration'] = 15.0
        features['hist_min_frustration'] = 5.0
        features['hist_total_duration'] = 600.0  # 10時間分
        features['hist_avg_duration'] = 60.0
        features['hist_activity_changes'] = 5

        # Fitbit統計特徴量（過去24時間の平均）
        # 注意: これらも固定値のため予測が偏ります
        fitbit_features = {
            'hist_lorenz_mean': 8000.0,
            'hist_lorenz_std': 1000.0,
            'hist_lorenz_min': 6000.0,
            'hist_lorenz_max': 10000.0,
            'hist_lorenz_median': 8000.0,
            'hist_lorenz_q25': 7000.0,
            'hist_lorenz_q75': 9000.0
        }
        features.update(fitbit_features)

        # 時間帯別特徴量（過去24時間）
        # これらも固定値
        for period in ['night', 'morning', 'afternoon', 'evening']:
            features[f'hist_{period}_avg_frustration'] = 10.0
            features[f'hist_{period}_duration'] = 150.0

        logger.warning(f"⚠️ 予測に固定値の特徴量を使用しています。活動: {activity_category}")

        return features

    def predict_with_history(self, activity_category: str, duration: int,
                            current_time: datetime, historical_data: pd.DataFrame) -> dict:
        """
        過去の履歴データを使用して予測（訓練時と同じ方法）

        Args:
            activity_category: 活動カテゴリ
            duration: 活動時間（分）
            current_time: 現在時刻
            historical_data: 過去のデータ（aggregate_fitbit_by_activityの出力）

        Returns:
            予測結果
        """
        try:
            if self.model is None:
                return self.predict_single_activity(activity_category, duration, current_time)

            if historical_data.empty or len(historical_data) < 5:
                logger.warning("履歴データが不足しているため、簡易予測を使用します")
                return self.predict_single_activity(activity_category, duration, current_time)

            # 過去24時間のデータを取得
            lookback_time = current_time - timedelta(hours=24)
            recent_data = historical_data[historical_data['Timestamp'] >= lookback_time]

            if recent_data.empty:
                # 全履歴データを使用
                recent_data = historical_data

            # 特徴量を構築（訓練時と同じ方法）
            features = {}

            # 現在の活動の基本特徴量
            features['current_hour'] = current_time.hour
            features['current_dayofweek'] = current_time.weekday()
            features['current_is_weekend'] = 1 if current_time.weekday() >= 5 else 0
            features['current_duration'] = duration

            # 活動カテゴリのエンコード
            if 'current_activity' in self.encoders:
                try:
                    features['current_activity'] = self.encoders['current_activity'].transform([activity_category])[0]
                except ValueError:
                    try:
                        features['current_activity'] = self.encoders['current_activity'].transform(['その他'])[0]
                    except:
                        features['current_activity'] = 0
            else:
                features['current_activity'] = 0

            # 過去24時間の統計特徴量（実データを使用）
            features['hist_avg_frustration'] = recent_data['NASA_F'].mean()
            features['hist_std_frustration'] = recent_data['NASA_F'].std() if len(recent_data) > 1 else 0
            features['hist_max_frustration'] = recent_data['NASA_F'].max()
            features['hist_min_frustration'] = recent_data['NASA_F'].min()
            features['hist_total_duration'] = recent_data['Duration'].sum()
            features['hist_avg_duration'] = recent_data['Duration'].mean()
            features['hist_activity_changes'] = recent_data['activity_change'].sum() if 'activity_change' in recent_data.columns else 0

            # Fitbit統計特徴量（実データを使用）
            fitbit_cols = ['lorenz_mean', 'lorenz_std', 'lorenz_min', 'lorenz_max',
                          'lorenz_median', 'lorenz_q25', 'lorenz_q75']
            for col in fitbit_cols:
                if col in recent_data.columns:
                    features[f'hist_{col}'] = recent_data[col].mean()
                else:
                    features[f'hist_{col}'] = 8000.0

            # 時間帯別特徴量（実データを使用）
            for start_hour, end_hour, label in [(0, 6, 'night'), (6, 12, 'morning'),
                                              (12, 18, 'afternoon'), (18, 24, 'evening')]:
                hour_data = recent_data[
                    (recent_data['hour'] >= start_hour) &
                    (recent_data['hour'] < end_hour)
                ] if 'hour' in recent_data.columns else pd.DataFrame()

                features[f'hist_{label}_avg_frustration'] = hour_data['NASA_F'].mean() if not hour_data.empty else 10.0
                features[f'hist_{label}_duration'] = hour_data['Duration'].sum() if not hour_data.empty else 0.0

            # NaN値を置換
            for key, value in features.items():
                if pd.isna(value):
                    if 'frustration' in key:
                        features[key] = 10.0
                    elif 'duration' in key:
                        features[key] = 0.0
                    elif 'lorenz' in key:
                        features[key] = 8000.0
                    else:
                        features[key] = 0.0

            # 予測実行
            pred_df = pd.DataFrame([features])

            # モデルの特徴量に合わせる
            for col in self.feature_columns:
                if col not in pred_df.columns:
                    pred_df[col] = 0.0

            pred_df = pred_df[self.feature_columns]
            prediction = self.model.predict(pred_df)[0]
            confidence = self.get_prediction_confidence(prediction, features)

            return {
                'predicted_frustration': float(prediction),
                'confidence': float(confidence),
                'activity_category': activity_category,
                'duration': duration,
                'timestamp': current_time,
                'features_used': len(self.feature_columns),
                'used_historical_data': True,
                'historical_records': len(recent_data)
            }

        except Exception as e:
            logger.error(f"履歴データを使った予測エラー: {e}")
            return self.predict_single_activity(activity_category, duration, current_time)

    def predict_single_activity(self, activity_category: str, duration: int = 60,
                               current_time: datetime = None) -> dict:
        """
        単一の活動に対してフラストレーション値を予測
        """
        try:
            if self.model is None:
                logger.warning("⚠️ モデルが訓練されていません。デフォルト値を返します。")
                return {
                    'predicted_frustration': 10.0,
                    'confidence': 0.0,
                    'error': 'モデルが訓練されていません',
                    'diagnosis': 'モデル未訓練: データが10件以上になると自動訓練されます'
                }

            # 特徴量作成
            features = self.create_features_for_new_activity(activity_category, duration, current_time)

            # DataFrameに変換
            feature_df = pd.DataFrame([features])

            # モデルで使用する特徴量に合わせる
            for col in self.feature_columns:
                if col not in feature_df.columns:
                    feature_df[col] = 0.0

            feature_df = feature_df[self.feature_columns]

            # 予測実行
            prediction = self.model.predict(feature_df)[0]
            confidence = self.get_prediction_confidence(prediction, features)

            # デバッグ情報: 予測値の多様性チェック
            diagnosis = None
            if hasattr(self, '_last_predictions'):
                self._last_predictions.append(prediction)
                if len(self._last_predictions) > 10:
                    self._last_predictions = self._last_predictions[-10:]

                # 直近10件の予測値の標準偏差をチェック
                if len(self._last_predictions) >= 5:
                    pred_std = np.std(self._last_predictions)
                    if pred_std < 0.5:
                        diagnosis = f"警告: 予測値の多様性が低い（標準偏差: {pred_std:.3f}）。データの分散不足またはモデルの問題の可能性"
                        logger.warning(f"⚠️ {diagnosis}")
            else:
                self._last_predictions = [prediction]

            result = {
                'predicted_frustration': float(prediction),
                'confidence': float(confidence),
                'activity_category': activity_category,
                'duration': duration,
                'timestamp': current_time or datetime.now(),
                'features_used': len(self.feature_columns)
            }

            if diagnosis:
                result['diagnosis'] = diagnosis

            return result
            
        except Exception as e:
            logger.error(f"単一活動予測エラー: {e}")
            return {
                'predicted_frustration': 10.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def predict_frustration_batch(self, activities: List[dict]) -> List[dict]:
        """
        複数の活動に対してフラストレーション値をバッチ予測
        
        Args:
            activities: 活動のリスト。各活動は{'sub_category': str, 'duration': int, 'timestamp': datetime}形式
            
        Returns:
            予測結果のリスト
        """
        try:
            if not activities:
                return []
                
            results = []
            for activity in activities:
                # 各活動に対して予測実行
                prediction_result = self.predict_single_activity(
                    activity_category=activity.get('sub_category', '不明'),
                    duration=activity.get('duration', 60),
                    current_time=activity.get('timestamp')
                )
                results.append(prediction_result)
                
            return results
            
        except Exception as e:
            logger.error(f"バッチ予測エラー: {e}")
            return []
    
    def get_prediction_confidence(self, prediction: float, features: dict) -> float:
        """
        予測の信頼度を計算
        """
        try:
            # 基本信頼度（予測値が妥当な範囲内かどうか）
            base_confidence = 0.7 if 1 <= prediction <= 20 else 0.3
            
            # 特徴量の完全性による調整
            feature_completeness = len([v for v in features.values() if v != 0]) / len(features)
            completeness_bonus = feature_completeness * 0.2
            
            # 時間帯による調整（通常の活動時間帯は信頼度が高い）
            hour = features.get('hour', 12)
            time_bonus = 0.1 if 6 <= hour <= 22 else 0.0
            
            # 最終信頼度計算
            confidence = min(0.95, base_confidence + completeness_bonus + time_bonus)
            
            return confidence
            
        except Exception as e:
            logger.error(f"信頼度計算エラー: {e}")
            return 0.5