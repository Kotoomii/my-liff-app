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
                start_time = activity['Timestamp']
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
            for hour_range, label in [(0, 6, 'night'), (6, 12, 'morning'), 
                                    (12, 18, 'afternoon'), (18, 24, 'evening')]:
                hour_data = historical_data[
                    (historical_data['hour'] >= hour_range[0]) & 
                    (historical_data['hour'] < hour_range[1])
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
    
    def walk_forward_validation_train(self, df_enhanced: pd.DataFrame) -> Dict:
        """
        Walk Forward Validationによる学習
        各行動変更タイミングで、過去24時間のデータで学習し、現在のフラストレーション値を予測
        """
        try:
            if len(df_enhanced) < 10:
                raise ValueError("Walk Forward Validationには最低10個のデータポイントが必要です")
            
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
            
            results = {
                'walk_forward_rmse': rmse,
                'walk_forward_mae': mae,
                'walk_forward_r2': r2,
                'total_predictions': len(predictions),
                'feature_importance': dict(zip(self.feature_columns, model.feature_importances_)),
                'prediction_history': training_results[-10:]  # 最新10件のみ返す
            }
            
            logger.info(f"Walk Forward Validation完了 - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")
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
                logger.info(f"モデルを読み込みました: {filepath}")
                return True
            else:
                logger.warning(f"モデルファイルが見つかりません: {filepath}")
                return False
        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
            return False