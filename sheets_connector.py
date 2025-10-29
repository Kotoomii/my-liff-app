"""
Googleスプレッドシート連携機能
"""

import pandas as pd
import logging
from typing import Optional, Dict, List
import gspread
from google.oauth2.service_account import Credentials
import os
import json
import tempfile
from datetime import datetime, timedelta
try:
    import openpyxl
except ImportError:
    openpyxl = None

from config import Config

logger = logging.getLogger(__name__)

class SheetsConnector:
    def __init__(self):
        self.config = Config()
        self.gc = None
        self.spreadsheet = None
        self.debug_mode = self._detect_debug_mode()
        self._initialize_client()

        # キャッシュ機能の初期化
        self._cache = {}
        self._cache_expiry = {}
        self._cache_duration_minutes = 5  # デフォルト5分間キャッシュ
        self._last_data_timestamp = {}  # 各シートの最終データタイムスタンプ
    
    def _get_cache_key(self, data_type: str, user_id: str) -> str:
        """キャッシュキーを生成"""
        return f"{data_type}_{user_id}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """キャッシュが有効かチェック"""
        if cache_key not in self._cache_expiry:
            return False
        return datetime.now() < self._cache_expiry[cache_key]

    def _get_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """キャッシュからデータを取得"""
        if self._is_cache_valid(cache_key):
            logger.info(f"キャッシュからデータを取得: {cache_key}")
            return self._cache[cache_key].copy()
        return None

    def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """データをキャッシュに保存"""
        self._cache[cache_key] = data.copy()
        self._cache_expiry[cache_key] = datetime.now() + timedelta(minutes=self._cache_duration_minutes)
        logger.info(f"データをキャッシュに保存: {cache_key} (有効期限: {self._cache_duration_minutes}分)")

    def _clear_cache(self, cache_key: str = None):
        """キャッシュをクリア"""
        if cache_key:
            if cache_key in self._cache:
                del self._cache[cache_key]
                del self._cache_expiry[cache_key]
                logger.info(f"キャッシュをクリア: {cache_key}")
        else:
            self._cache.clear()
            self._cache_expiry.clear()
            logger.info("すべてのキャッシュをクリア")

    def has_new_data(self, user_id: str = "default") -> bool:
        """
        新しいデータが追加されたかチェック
        最終データのタイムスタンプを比較して判定
        """
        try:
            # 現在のデータを取得（キャッシュを使わない）
            activity_data = self._get_activity_data_from_sheets(user_id) if not self.debug_mode else self._get_activity_data_from_excel(user_id)

            if activity_data.empty:
                return False

            # 最新のタイムスタンプを取得
            if 'Timestamp' not in activity_data.columns:
                return False

            latest_timestamp = activity_data['Timestamp'].max()
            cache_key = self._get_cache_key('activity', user_id)

            # 前回の最終タイムスタンプと比較
            if cache_key in self._last_data_timestamp:
                if latest_timestamp > self._last_data_timestamp[cache_key]:
                    logger.info(f"新しいデータを検知: {user_id} (最新: {latest_timestamp})")
                    self._last_data_timestamp[cache_key] = latest_timestamp
                    # 新しいデータがあればキャッシュをクリア
                    self._clear_cache(cache_key)
                    return True
                return False
            else:
                # 初回は新データとして扱わない
                self._last_data_timestamp[cache_key] = latest_timestamp
                return False

        except Exception as e:
            logger.error(f"新データ検知エラー: {e}")
            return False

    def _detect_debug_mode(self) -> bool:
        """デバッグモードを検出（ローカルExcelファイルの存在確認）"""
        excel_file_path = os.path.join(os.path.dirname(__file__), 'data', '確認用.xlsx')
        debug_mode = os.path.exists(excel_file_path) and openpyxl is not None

        if debug_mode:
            logger.info(f"デバッグモード: ローカルExcelファイルを使用 ({excel_file_path})")
        else:
            logger.info("プロダクションモード: Google Sheetsを使用")

        return debug_mode
    
    def _initialize_client(self):
        """Google Sheets APIクライアントを初期化"""
        try:
            # サービスアカウント認証
            scope = [
                'https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive'
            ]
            
            # 1. 環境変数からJSONを取得 (Secret Manager経由)
            credentials_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
            
            if credentials_json:
                # Secret ManagerからのJSON文字列を使用
                credentials_info = json.loads(credentials_json)
                creds = Credentials.from_service_account_info(credentials_info, scopes=scope)
                logger.info("Secret ManagerからGoogle認証情報を取得")
            else:
                # 2. ファイルパスから読み込み (ローカル開発用)
                credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
                if credentials_path and os.path.exists(credentials_path):
                    creds = Credentials.from_service_account_file(credentials_path, scopes=scope)
                    logger.info("ファイルからGoogle認証情報を取得")
                else:
                    # 3. デフォルト認証 (GCP環境)
                    try:
                        from google.auth import default
                        creds, project = default(scopes=scope)
                        logger.info("デフォルト認証を使用")
                    except Exception as e:
                        logger.warning(f"認証情報の取得に失敗: {e}")
                        return
            
            self.gc = gspread.authorize(creds)
            self.spreadsheet = self.gc.open_by_key(self.config.SPREADSHEET_ID)
            logger.info("Google Sheetsクライアント初期化完了")
            
        except Exception as e:
            logger.error(f"Google Sheetsクライアント初期化エラー: {e}")
            self.gc = None
    
    def _find_worksheet_by_pattern(self, pattern: str) -> Optional[gspread.Worksheet]:
        """パターンに一致するワークシートを検索"""
        try:
            if not self.spreadsheet:
                return None
                
            worksheets = self.spreadsheet.worksheets()
            for ws in worksheets:
                if pattern in ws.title:
                    return ws
            
            logger.warning(f"パターン '{pattern}' に一致するワークシートが見つかりません")
            return None
            
        except Exception as e:
            logger.error(f"ワークシート検索エラー: {e}")
            return None
    
    def _find_worksheet_by_exact_name(self, sheet_name: str) -> Optional[gspread.Worksheet]:
        """正確なシート名に一致するワークシートを検索"""
        try:
            if not self.spreadsheet:
                return None
                
            worksheets = self.spreadsheet.worksheets()
            for ws in worksheets:
                if ws.title == sheet_name:
                    return ws
            
            logger.warning(f"シート '{sheet_name}' が見つかりません")
            return None
            
        except Exception as e:
            logger.error(f"ワークシート検索エラー: {e}")
            return None
    
    def _get_user_sheet_config(self, user_id: str) -> Dict:
        """ユーザー設定を取得（Config.pyから）"""
        return self.config.get_user_config(user_id)
    
    def get_activity_data(self, user_id: str = "default", use_cache: bool = True) -> pd.DataFrame:
        """
        活動データを取得（デバッグモード: Excelファイル、プロダクション: Google Sheets）
        キャッシュ機能付き
        """
        try:
            cache_key = self._get_cache_key('activity', user_id)

            # キャッシュチェック
            if use_cache:
                cached_data = self._get_from_cache(cache_key)
                if cached_data is not None:
                    return cached_data

            # デバッグモード: ローカルExcelファイルから読み込み
            if self.debug_mode:
                data = self._get_activity_data_from_excel(user_id)
            else:
                # プロダクションモード: Google Sheetsから読み込み
                data = self._get_activity_data_from_sheets(user_id)

            # キャッシュに保存
            if not data.empty and use_cache:
                self._save_to_cache(cache_key, data)

            return data

        except Exception as e:
            logger.error(f"活動データ取得エラー: {e}")
            return pd.DataFrame()
    
    def _get_activity_data_from_excel(self, user_id: str = "default") -> pd.DataFrame:
        """ExcelファイルからJA動データを取得"""
        try:
            excel_file_path = os.path.join(os.path.dirname(__file__), 'data', '確認用.xlsx')
            
            if not os.path.exists(excel_file_path):
                logger.warning(f"Excelファイルが見つかりません: {excel_file_path}")
                return pd.DataFrame()
            
            # ユーザー設定から活動データシート名を取得
            user_config = self._get_user_sheet_config(user_id)
            activity_sheet_name = user_config.get('activity_sheet', user_id)
            
            # Excelファイルから指定シートを読み込み
            df = pd.read_excel(excel_file_path, sheet_name=activity_sheet_name)
            
            if df.empty:
                logger.warning(f"活動データが空です (Excel: {activity_sheet_name})")
                return pd.DataFrame()
            
            # データ型変換
            if 'Timestamp' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            
            # NASA-TLX列の数値変換（1-20スケール用）
            nasa_cols = [col for col in self.config.NASA_DIMENSIONS if col in df.columns]
            for col in nasa_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(10)  # 1-20スケール用のデフォルト値
            
            logger.info(f"活動データ取得完了 (Excel: {activity_sheet_name}): {len(df)} 行")
            return df
            
        except Exception as e:
            logger.error(f"Excel活動データ取得エラー: {e}")
            return pd.DataFrame()
    
    def _get_activity_data_from_sheets(self, user_id: str = "default") -> pd.DataFrame:
        """Google Sheetsから活動データを取得"""
        try:
            if not self.gc:
                logger.warning("Google Sheetsクライアントが初期化されていません")
                return pd.DataFrame()
            
            # ユーザー設定から活動データシート名を取得
            user_config = self._get_user_sheet_config(user_id)
            activity_sheet_name = user_config.get('activity_sheet', user_id)
            
            # ユーザー固有の活動データシートを検索
            worksheet = self._find_worksheet_by_exact_name(activity_sheet_name)
            
            if not worksheet:
                logger.warning(f"活動データシート '{activity_sheet_name}' が見つかりません")
                return pd.DataFrame()
            
            # データ取得
            data = worksheet.get_all_records()
            if not data:
                logger.warning("活動データが空です")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            
            # データ型変換
            if 'Timestamp' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            
            # NASA-TLX列の数値変換
            nasa_cols = [col for col in self.config.NASA_DIMENSIONS if col in df.columns]
            for col in nasa_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(10)  # 1-20スケール用のデフォルト値
            
            logger.info(f"活動データ取得完了 (シート: {activity_sheet_name}): {len(df)} 行")
            return df
            
        except Exception as e:
            logger.error(f"Google Sheets活動データ取得エラー: {e}")
            return pd.DataFrame()
    
    def get_fitbit_data(self, user_id: str = "default", use_cache: bool = True) -> pd.DataFrame:
        """
        生体データ（Fitbitデータ）を取得（デバッグモード: Excelファイル、プロダクション: Google Sheets）
        キャッシュ機能付き
        """
        try:
            cache_key = self._get_cache_key('fitbit', user_id)

            # キャッシュチェック
            if use_cache:
                cached_data = self._get_from_cache(cache_key)
                if cached_data is not None:
                    return cached_data

            # デバッグモード: ローカルExcelファイルから読み込み
            if self.debug_mode:
                data = self._get_fitbit_data_from_excel(user_id)
            else:
                # プロダクションモード: Google Sheetsから読み込み
                data = self._get_fitbit_data_from_sheets(user_id)

            # キャッシュに保存
            if not data.empty and use_cache:
                self._save_to_cache(cache_key, data)

            return data

        except Exception as e:
            logger.error(f"Fitbitデータ取得エラー: {e}")
            return pd.DataFrame()
    
    def _get_fitbit_data_from_excel(self, user_id: str = "default") -> pd.DataFrame:
        """ExcelファイルからFitbitデータを取得"""
        try:
            excel_file_path = os.path.join(os.path.dirname(__file__), 'data', '確認用.xlsx')
            
            if not os.path.exists(excel_file_path):
                logger.warning(f"Excelファイルが見つかりません: {excel_file_path}")
                return pd.DataFrame()
            
            # ユーザー設定からFitbitデータシート名を取得
            user_config = self._get_user_sheet_config(user_id)
            fitbit_sheet_name = user_config.get('fitbit_sheet', f'kotoomi_Fitbit-data-{user_id}')
            
            # Excelファイルから指定シートを読み込み
            df = pd.read_excel(excel_file_path, sheet_name=fitbit_sheet_name)
            
            if df.empty:
                logger.warning(f"Fitbitデータが空です (Excel: {fitbit_sheet_name})")
                return pd.DataFrame()
            
            # データ型変換
            if 'Timestamp' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            if 'Lorenz_Area' in df.columns:
                df['Lorenz_Area'] = pd.to_numeric(df['Lorenz_Area'], errors='coerce').fillna(8000)
            if 'SDNN' in df.columns:
                df['SDNN'] = pd.to_numeric(df['SDNN'], errors='coerce').fillna(50)
            if 'data_points' in df.columns:
                df['data_points'] = pd.to_numeric(df['data_points'], errors='coerce').fillna(100)
            
            logger.info(f"Fitbitデータ取得完了 (Excel: {fitbit_sheet_name}): {len(df)} 行")
            return df
            
        except Exception as e:
            logger.error(f"Excel Fitbitデータ取得エラー: {e}")
            return pd.DataFrame()
    
    def _get_fitbit_data_from_sheets(self, user_id: str = "default") -> pd.DataFrame:
        """Google SheetsからFitbitデータを取得"""
        try:
            if not self.gc:
                logger.warning("Google Sheetsクライアントが初期化されていません")
                return pd.DataFrame()
            
            # ユーザー設定からFitbitデータシート名を取得
            user_config = self._get_user_sheet_config(user_id)
            fitbit_sheet_name = user_config.get('fitbit_sheet', f'kotoomi_Fitbit-data-{user_id}')
            
            # ユーザー固有のFitbitデータシートを検索
            worksheet = self._find_worksheet_by_exact_name(fitbit_sheet_name)
            
            if not worksheet:
                logger.warning(f"Fitbitデータシート '{fitbit_sheet_name}' が見つかりません")
                return pd.DataFrame()
            
            # データ取得
            data = worksheet.get_all_records()
            if not data:
                logger.warning("Fitbitデータが空です")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            
            # データ型変換
            if 'Timestamp' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            if 'Lorenz_Area' in df.columns:
                df['Lorenz_Area'] = pd.to_numeric(df['Lorenz_Area'], errors='coerce').fillna(8000)
            if 'SDNN' in df.columns:
                df['SDNN'] = pd.to_numeric(df['SDNN'], errors='coerce').fillna(50)
            if 'data_points' in df.columns:
                df['data_points'] = pd.to_numeric(df['data_points'], errors='coerce').fillna(100)
            
            logger.info(f"Fitbitデータ取得完了 (シート: {fitbit_sheet_name}): {len(df)} 行")
            return df
            
        except Exception as e:
            logger.error(f"Google Sheets Fitbitデータ取得エラー: {e}")
            return pd.DataFrame()
    
    def get_fixed_plans(self, user_id: str = "default") -> pd.DataFrame:
        """
        固定予定データ（FIXED_PLANS）を取得（Google Sheetsのみ対応）
        """
        try:
            if not self.gc:
                logger.warning("Google Sheetsクライアントが初期化されていません")
                return pd.DataFrame()
            
            # FIXED_PLANSシートを検索
            worksheet = self._find_worksheet_by_pattern(self.config.FIXED_PLANS_SHEET)
            
            if not worksheet:
                logger.warning("FIXED_PLANSシートが見つかりません")
                return pd.DataFrame()
            
            # データ取得
            data = worksheet.get_all_records()
            if not data:
                logger.warning("FIXED_PLANSデータが空です")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            
            # データ型変換
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            if 'StartTime' in df.columns:
                df['StartTime'] = pd.to_datetime(df['StartTime'], format='%H:%M', errors='coerce').dt.time
            if 'EndTime' in df.columns:
                df['EndTime'] = pd.to_datetime(df['EndTime'], format='%H:%M', errors='coerce').dt.time
            
            # ユーザーフィルタリング（もしUserID列がある場合）
            if 'UserID' in df.columns:
                df = df[df['UserID'] == user_id]
            
            logger.info(f"FIXED_PLANSデータ取得完了: {len(df)} 行")
            return df
            
        except Exception as e:
            logger.error(f"FIXED_PLANSデータ取得エラー: {e}")
            return pd.DataFrame()
    
    def _get_dummy_activity_data(self) -> pd.DataFrame:
        """ダミーの活動データを生成"""
        from datetime import datetime, timedelta
        import numpy as np
        
        # 過去7日分のダミーデータを生成
        data = []
        base_time = datetime.now() - timedelta(days=7)
        
        activities = {
            '睡眠': {'frustration': 3},
            '食事': {'frustration': 5}, 
            '仕事': {'frustration': 15},
            '休憩': {'frustration': 2},
            '運動': {'frustration': 7}
        }
        
        for day in range(7):
            for hour in range(0, 24, 3):  # 3時間おき
                timestamp = base_time + timedelta(days=day, hours=hour)
                activity = np.random.choice(list(activities.keys()))
                base_frustration = activities[activity]['frustration']
                
                # NASA-TLXスコアを生成（1-20スケール）
                nasa_scores = np.random.normal(base_frustration, 3, 6).clip(1, 20).astype(int)
                
                data.append({
                    'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'CatSub': activity,
                    'Duration': 180,  # 3時間
                    'NASA_M': nasa_scores[0],
                    'NASA_P': nasa_scores[1], 
                    'NASA_T': nasa_scores[2],
                    'NASA_O': nasa_scores[3],
                    'NASA_E': nasa_scores[4],
                    'NASA_F': nasa_scores[5]
                })
        
        df = pd.DataFrame(data)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        logger.info("ダミー活動データを生成しました")
        return df
    
    def _get_dummy_fitbit_data(self) -> pd.DataFrame:
        """ダミーのFitbitデータを生成"""
        from datetime import datetime, timedelta
        import numpy as np
        
        data = []
        base_time = datetime.now() - timedelta(days=7)
        
        for day in range(7):
            for hour in range(24):
                for minute in [0, 15, 30, 45]:  # 15分おき
                    timestamp = base_time + timedelta(days=day, hours=hour, minutes=minute)
                    lorenz_area = np.random.normal(8000, 1000)  # ローレンツプロット面積
                    
                    data.append({
                        'Timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        'Lorenz_Area': max(lorenz_area, 500)  # 最低値を設定
                    })
        
        df = pd.DataFrame(data)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        logger.info("ダミーFitbitデータを生成しました")
        return df
    
    def _get_dummy_fixed_plans(self) -> pd.DataFrame:
        """ダミーの固定予定データを生成"""
        from datetime import datetime, timedelta
        import numpy as np
        
        data = []
        base_date = datetime.now().date()
        
        # 今日から1週間分の固定予定を生成
        fixed_activities = [
            {'activity': '朝食', 'start': '07:30', 'end': '08:00'},
            {'activity': '昼食', 'start': '12:00', 'end': '13:00'},
            {'activity': '夕食', 'start': '18:30', 'end': '19:30'},
            {'activity': '会議', 'start': '10:00', 'end': '11:00'},
            {'activity': '定期健診', 'start': '14:00', 'end': '16:00'}
        ]
        
        for day in range(7):
            current_date = base_date + timedelta(days=day)
            
            # 毎日の食事は固定
            for meal in fixed_activities[:3]:
                data.append({
                    'Date': current_date.strftime('%Y-%m-%d'),
                    'Activity': meal['activity'],
                    'StartTime': meal['start'],
                    'EndTime': meal['end'],
                    'UserID': 'default',
                    'Fixed': 'Yes'
                })
            
            # 平日のみ会議
            if current_date.weekday() < 5:  # 月-金
                data.append({
                    'Date': current_date.strftime('%Y-%m-%d'),
                    'Activity': fixed_activities[3]['activity'],
                    'StartTime': fixed_activities[3]['start'],
                    'EndTime': fixed_activities[3]['end'],
                    'UserID': 'default',
                    'Fixed': 'Yes'
                })
            
            # 特定の日に健診（例：3日後）
            if day == 3:
                data.append({
                    'Date': current_date.strftime('%Y-%m-%d'),
                    'Activity': fixed_activities[4]['activity'],
                    'StartTime': fixed_activities[4]['start'],
                    'EndTime': fixed_activities[4]['end'],
                    'UserID': 'default',
                    'Fixed': 'Yes'
                })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df['Date'] = pd.to_datetime(df['Date'])
            df['StartTime'] = pd.to_datetime(df['StartTime'], format='%H:%M', errors='coerce').dt.time
            df['EndTime'] = pd.to_datetime(df['EndTime'], format='%H:%M', errors='coerce').dt.time
        
        logger.info("ダミー固定予定データを生成しました")
        return df
    
    def is_prediction_duplicate(self, user_id: str, activity_timestamp: str) -> bool:
        """
        指定されたユーザーIDと活動タイムスタンプの組み合わせで
        既に予測データが存在するかチェックする
        
        Args:
            user_id: ユーザーID
            activity_timestamp: 活動データのタイムスタンプ
            
        Returns:
            bool: 重複している場合True、そうでなければFalse
        """
        try:
            if not self.gc:
                logger.warning("Google Sheetsクライアントが初期化されていません")
                return False

            # PREDICTION_DATAシートを取得
            sheet_name = "PREDICTION_DATA"
            worksheet = self._find_worksheet_by_exact_name(sheet_name)
            
            if not worksheet:
                # シートが存在しない場合は重複なし
                return False
            
            # 全データを取得
            records = worksheet.get_all_records()
            
            if not records:
                # データが存在しない場合は重複なし
                return False
            
            # 同じユーザーIDで、活動タイムスタンプに基づく予測データが既に存在するかチェック
            for record in records:
                if record.get('UserID') == user_id:
                    # 活動タイムスタンプが一致するかチェック
                    # activity_timestampとの比較（Notes欄にTimestamp情報がある場合もある）
                    notes = record.get('Notes', '')
                    if activity_timestamp in notes:
                        logger.info(f"重複する予測データを発見: UserID={user_id}, ActivityTimestamp={activity_timestamp}")
                        return True
                        
                    # 予測タイムスタンプが活動タイムスタンプと近い場合も重複とみなす
                    prediction_timestamp = record.get('Timestamp', '')
                    if prediction_timestamp and activity_timestamp:
                        try:
                            from datetime import datetime, timedelta
                            pred_time = datetime.fromisoformat(prediction_timestamp.replace('Z', '+00:00') if 'Z' in prediction_timestamp else prediction_timestamp)
                            activity_time = datetime.fromisoformat(activity_timestamp.replace('Z', '+00:00') if 'Z' in activity_timestamp else activity_timestamp)
                            
                            # 5分以内の差の場合は重複とみなす
                            if abs((pred_time - activity_time).total_seconds()) < 300:  # 5分 = 300秒
                                logger.info(f"時間が近い予測データを発見: UserID={user_id}, 差={abs((pred_time - activity_time).total_seconds())}秒")
                                return True
                        except (ValueError, TypeError) as e:
                            # タイムスタンプのパースエラーは無視
                            logger.debug(f"タイムスタンプパースエラー: {e}")
                            continue
            
            return False
            
        except Exception as e:
            logger.error(f"予測データ重複チェックエラー: {e}")
            # エラーの場合は安全のため重複なしとして処理を続行
            return False

    def save_prediction_data(self, prediction_data: Dict) -> bool:
        """
        予測結果をスプレッドシートに保存（ユーザー別シート）
        タイムスタンプによる重複をチェックし、既存行があれば更新、なければ追加

        新構造:
        - Timestamp, Activity, Duration, ActualFrustration, PredictedFrustration,
          PredictionError, Confidence, Notes
        """
        try:
            if not self.gc:
                logger.warning("Google Sheetsクライアントが初期化されていません")
                return False

            # ユーザー別のPREDICTION_DATAシート名を生成
            user_id = prediction_data.get('user_id', 'default')
            sheet_name = f"PREDICTION_DATA_{user_id}"
            worksheet = self._find_worksheet_by_exact_name(sheet_name)

            if not worksheet:
                # シートが存在しない場合は作成
                worksheet = self.spreadsheet.add_worksheet(
                    title=sheet_name,
                    rows="10000",
                    cols="8"
                )
                # ヘッダー行を追加（新構造）
                headers = [
                    'Timestamp', 'Activity', 'Duration', 'ActualFrustration',
                    'PredictedFrustration', 'PredictionError', 'Confidence', 'Notes'
                ]
                worksheet.update('A1:H1', [headers])  # update()を使用して確実にA1から書き込む
                logger.info(f"PREDICTION_DATAシートを作成しました: {sheet_name}")

            # データ行を準備（新構造）
            predicted_frustration = prediction_data.get('predicted_frustration', 0)
            actual_frustration = prediction_data.get('actual_frustration')

            # 予測誤差を計算（実測値がある場合のみ）
            prediction_error = None
            if actual_frustration is not None and predicted_frustration is not None:
                prediction_error = abs(predicted_frustration - actual_frustration)

            # Notes欄
            notes = prediction_data.get('notes', '')
            source = prediction_data.get('source', 'manual')

            # タイムスタンプを取得
            timestamp = prediction_data.get('timestamp', datetime.now().isoformat())

            # 新構造のrow_data: Timestamp, Activity, Duration, ActualFrustration,
            #                    PredictedFrustration, PredictionError, Confidence, Notes
            row_data = [
                timestamp,
                prediction_data.get('activity', 'unknown'),
                prediction_data.get('duration', 0),
                round(actual_frustration, 2) if actual_frustration is not None else '',
                round(predicted_frustration, 2),
                round(prediction_error, 2) if prediction_error is not None else '',
                round(prediction_data.get('confidence', 0), 3),
                f"{source}: {notes}" if notes else source
            ]

            # 重複チェック: 既存のデータから同じタイムスタンプを探す
            try:
                existing_data = worksheet.get_all_values()
                existing_row_index = None

                # ヘッダー行をスキップして検索（1行目はヘッダー）
                for i, row in enumerate(existing_data[1:], start=2):  # start=2 (行番号は1から始まり、ヘッダーが1行目)
                    if len(row) > 0 and row[0] == timestamp:  # Timestamp列（0番目）で比較
                        existing_row_index = i
                        logger.info(f"重複するタイムスタンプを発見: {timestamp} (行: {i})")
                        break

                if existing_row_index:
                    # 既存行を更新（新構造: A-H列）
                    cell_range = f'A{existing_row_index}:H{existing_row_index}'
                    worksheet.update(cell_range, [row_data])
                    logger.info(f"予測データを更新しました: {prediction_data.get('activity')} (予測値: {predicted_frustration:.2f}, 実測値: {actual_frustration if actual_frustration else 'N/A'}, 行: {existing_row_index})")
                else:
                    # 新規行を追加
                    worksheet.append_row(row_data)
                    logger.info(f"予測データを保存しました: {prediction_data.get('activity')} (予測値: {predicted_frustration:.2f}, 実測値: {actual_frustration if actual_frustration else 'N/A'})")

            except Exception as duplicate_check_error:
                # 重複チェックに失敗した場合は通常の追加処理を実行
                logger.warning(f"重複チェックに失敗、新規追加します: {duplicate_check_error}")
                worksheet.append_row(row_data)
                logger.info(f"予測データを保存しました: {prediction_data.get('activity')} (予測値: {predicted_frustration:.2f})")

            return True

        except Exception as e:
            logger.error(f"予測データ保存エラー: {e}")
            return False

    def save_daily_summary(self, user_id: str, date: str, summary_data: Dict) -> bool:
        """
        日次サマリーデータをスプレッドシートに保存
        1日の平均フラストレーション値、最小値、最大値、活動数を記録
        """
        try:
            if not self.gc:
                logger.warning("Google Sheetsクライアントが初期化されていません")
                return False

            # ユーザー別のDAILY_SUMMARYシート名を生成
            sheet_name = f"DAILY_SUMMARY_{user_id}"
            worksheet = self._find_worksheet_by_exact_name(sheet_name)

            if not worksheet:
                # シートが存在しない場合は作成
                worksheet = self.spreadsheet.add_worksheet(
                    title=sheet_name,
                    rows="1000",
                    cols="8"
                )
                # ヘッダー行を追加
                headers = [
                    'Date', 'AvgFrustration', 'MinFrustration', 'MaxFrustration',
                    'ActivityCount', 'TotalDuration', 'UniqueActivities', 'Notes'
                ]
                worksheet.update('A1:H1', [headers])  # update()を使用して確実にA1から書き込む
                logger.info(f"DAILY_SUMMARYシートを作成しました: {sheet_name}")

            # データ行を準備
            avg_frustration = summary_data.get('avg_frustration', 0)
            min_frustration = summary_data.get('min_frustration', 0)
            max_frustration = summary_data.get('max_frustration', 0)
            activity_count = summary_data.get('activity_count', 0)
            total_duration = summary_data.get('total_duration', 0)
            unique_activities = summary_data.get('unique_activities', 0)
            notes = summary_data.get('notes', '')

            row_data = [
                date,
                round(avg_frustration, 2),
                round(min_frustration, 2),
                round(max_frustration, 2),
                activity_count,
                total_duration,
                unique_activities,
                notes
            ]

            # 重複チェック: 同じ日付のデータが既に存在するかチェック
            try:
                existing_data = worksheet.get_all_values()
                existing_row_index = None

                # ヘッダー行をスキップして検索
                for i, row in enumerate(existing_data[1:], start=2):
                    if len(row) > 0 and row[0] == date:  # Date列（0番目）で比較
                        existing_row_index = i
                        logger.info(f"既存の日次サマリーを発見: {date} (行: {i})")
                        break

                if existing_row_index:
                    # 既存行を更新
                    cell_range = f'A{existing_row_index}:H{existing_row_index}'
                    worksheet.update(cell_range, [row_data])
                    logger.info(f"日次サマリーを更新しました: {user_id}, {date}, 平均: {avg_frustration:.2f}")
                else:
                    # 新規行を追加
                    worksheet.append_row(row_data)
                    logger.info(f"日次サマリーを保存しました: {user_id}, {date}, 平均: {avg_frustration:.2f}")

            except Exception as duplicate_check_error:
                # 重複チェックに失敗した場合は通常の追加処理を実行
                logger.warning(f"重複チェックに失敗、新規追加します: {duplicate_check_error}")
                worksheet.append_row(row_data)
                logger.info(f"日次サマリーを保存しました: {user_id}, {date}, 平均: {avg_frustration:.2f}")

            return True

        except Exception as e:
            logger.error(f"日次サマリー保存エラー: {e}")
            return False

    def save_workload_data(self, user_id: str, date: str, workload_data: Dict) -> bool:
        """
        負荷データをスプレッドシートに保存
        """
        try:
            if not self.gc:
                logger.warning("Google Sheetsクライアントが初期化されていません")
                return False
            
            # WORKLOAD_DATAシートを取得または作成
            worksheet = self._find_worksheet_by_pattern(self.config.WORKLOAD_DATA_SHEET)
            
            if not worksheet:
                # シートが存在しない場合は作成
                worksheet = self.spreadsheet.add_worksheet(
                    title=self.config.WORKLOAD_DATA_SHEET,
                    rows="1000",
                    cols="20"
                )
                # ヘッダー行を追加
                headers = [
                    'Date', 'UserID', 'AvgStress', 'TotalWorkload', 'HighStressHours',
                    'MaxNASAItem', 'StressTolerance', 'ActivityBalance', 'RecoveryEfficiency',
                    'NASA_M_Avg', 'NASA_P_Avg', 'NASA_T_Avg', 'NASA_O_Avg', 'NASA_E_Avg', 'NASA_F_Avg',
                    'Timestamp'
                ]
                worksheet.append_row(headers)
                logger.info("WORKLOAD_DATAシートを作成しました")
            
            # データ行を準備
            row_data = [
                date,
                user_id,
                workload_data.get('avg_stress', 0),
                workload_data.get('total_workload', 0),
                workload_data.get('high_stress_hours', 0),
                workload_data.get('max_nasa_item', ''),
                workload_data.get('stress_tolerance', 0),
                workload_data.get('activity_balance', 0),
                workload_data.get('recovery_efficiency', 0),
                workload_data.get('nasa_averages', {}).get('NASA_M', 0),
                workload_data.get('nasa_averages', {}).get('NASA_P', 0),
                workload_data.get('nasa_averages', {}).get('NASA_T', 0),
                workload_data.get('nasa_averages', {}).get('NASA_O', 0),
                workload_data.get('nasa_averages', {}).get('NASA_E', 0),
                workload_data.get('nasa_averages', {}).get('NASA_F', 0),
                datetime.now().isoformat()
            ]
            
            # 既存データのチェック（同日同ユーザーのデータがあれば更新）
            existing_data = worksheet.get_all_records()
            existing_row = None
            
            for i, row in enumerate(existing_data, start=2):  # ヘッダー行を考慮してstart=2
                if row.get('Date') == date and row.get('UserID') == user_id:
                    existing_row = i
                    break
            
            if existing_row:
                # 既存行を更新
                worksheet.update(f'A{existing_row}:P{existing_row}', [row_data])
                logger.info(f"負荷データを更新しました: {user_id}, {date}")
            else:
                # 新規行を追加
                worksheet.append_row(row_data)
                logger.info(f"負荷データを保存しました: {user_id}, {date}")
            
            return True
            
        except Exception as e:
            logger.error(f"負荷データ保存エラー: {e}")
            return False
    
    def get_workload_data(self, user_id: str = "default", date: str = None) -> pd.DataFrame:
        """
        保存された負荷データを取得
        """
        try:
            if not self.gc:
                logger.warning("Google Sheetsクライアントが初期化されていません")
                return pd.DataFrame()
            
            worksheet = self._find_worksheet_by_pattern(self.config.WORKLOAD_DATA_SHEET)
            
            if not worksheet:
                logger.warning("WORKLOAD_DATAシートが見つかりません")
                return pd.DataFrame()
            
            # データ取得
            data = worksheet.get_all_records()
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            
            # データ型変換
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # フィルタリング
            if user_id != "all":
                df = df[df['UserID'] == user_id]
            
            if date:
                target_date = pd.to_datetime(date).date()
                df = df[df['Date'].dt.date == target_date]
            
            logger.info(f"負荷データ取得完了: {len(df)} 行")
            return df
            
        except Exception as e:
            logger.error(f"負荷データ取得エラー: {e}")
            return pd.DataFrame()
    
    def recreate_prediction_sheet(self, user_id: str = "default") -> bool:
        """
        PREDICTION_DATAシートを削除して再作成する
        古いシートをバックアップしてから新しい構造で作り直す
        """
        try:
            if not self.gc:
                logger.warning("Google Sheetsクライアントが初期化されていません")
                return False

            sheet_name = f"PREDICTION_DATA_{user_id}"
            worksheet = self._find_worksheet_by_exact_name(sheet_name)

            if worksheet:
                # 既存シートを削除
                self.spreadsheet.del_worksheet(worksheet)
                logger.info(f"既存のPREDICTION_DATAシートを削除しました: {sheet_name}")

            # 新しいシートを作成
            new_worksheet = self.spreadsheet.add_worksheet(
                title=sheet_name,
                rows="10000",
                cols="8"
            )

            # ヘッダー行を追加
            headers = [
                'Timestamp', 'Activity', 'Duration', 'ActualFrustration',
                'PredictedFrustration', 'PredictionError', 'Confidence', 'Notes'
            ]
            new_worksheet.update('A1:H1', [headers])
            logger.info(f"新しいPREDICTION_DATAシートを作成しました: {sheet_name}")

            return True

        except Exception as e:
            logger.error(f"PREDICTION_DATAシート再作成エラー: {e}")
            return False

    def recreate_daily_summary_sheet(self, user_id: str = "default") -> bool:
        """
        DAILY_SUMMARYシートを削除して再作成する
        """
        try:
            if not self.gc:
                logger.warning("Google Sheetsクライアントが初期化されていません")
                return False

            sheet_name = f"DAILY_SUMMARY_{user_id}"
            worksheet = self._find_worksheet_by_exact_name(sheet_name)

            if worksheet:
                # 既存シートを削除
                self.spreadsheet.del_worksheet(worksheet)
                logger.info(f"既存のDAILY_SUMMARYシートを削除しました: {sheet_name}")

            # 新しいシートを作成
            new_worksheet = self.spreadsheet.add_worksheet(
                title=sheet_name,
                rows="1000",
                cols="8"
            )

            # ヘッダー行を追加
            headers = [
                'Date', 'AvgFrustration', 'MinFrustration', 'MaxFrustration',
                'ActivityCount', 'TotalDuration', 'UniqueActivities', 'Notes'
            ]
            new_worksheet.update('A1:H1', [headers])
            logger.info(f"新しいDAILY_SUMMARYシートを作成しました: {sheet_name}")

            return True

        except Exception as e:
            logger.error(f"DAILY_SUMMARYシート再作成エラー: {e}")
            return False

    def test_connection(self) -> Dict:
        """接続テスト"""
        result = {
            'status': 'success',
            'spreadsheet_id': self.config.SPREADSHEET_ID,
            'worksheets': [],
            'activity_sheet_found': False,
            'fitbit_sheet_found': False
        }
        
        try:
            if not self.gc or not self.spreadsheet:
                result['status'] = 'error'
                result['error'] = 'Google Sheetsクライアントが初期化されていません'
                return result
            
            # ワークシート一覧取得
            worksheets = self.spreadsheet.worksheets()
            result['worksheets'] = [ws.title for ws in worksheets]
            
            # 目的のシートが存在するか確認
            activity_ws = self._find_worksheet_by_pattern(self.config.ACTIVITY_SHEET_PATTERN)
            fitbit_ws = self._find_worksheet_by_pattern(self.config.FITBIT_SHEET_PATTERN)
            
            result['activity_sheet_found'] = activity_ws is not None
            result['fitbit_sheet_found'] = fitbit_ws is not None
            
            if activity_ws:
                result['activity_sheet_name'] = activity_ws.title
            if fitbit_ws:
                result['fitbit_sheet_name'] = fitbit_ws.title
                
            logger.info("Google Sheets接続テスト完了")

        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            logger.error(f"接続テストエラー: {e}")

        return result

    def get_hourly_log(self, user_id: str, date: str) -> pd.DataFrame:
        """
        指定日のHourly Logデータを取得（予測結果のキャッシュとして使用）

        Args:
            user_id: ユーザーID
            date: 日付（YYYY-MM-DD形式）

        Returns:
            DataFrame: 日付, 時刻, 活動名, 実測NASA_F, 予測NASA_F, 誤差(MAE)
        """
        try:
            if not self.gc:
                logger.warning("Google Sheetsクライアントが初期化されていません")
                return pd.DataFrame()

            sheet_name = f"{user_id}_Hourly_Log"
            worksheet = self._find_worksheet_by_exact_name(sheet_name)

            if not worksheet:
                logger.info(f"Hourly Logシートが存在しません: {sheet_name}")
                return pd.DataFrame()

            # シートの全データを取得
            all_values = worksheet.get_all_values()

            if len(all_values) <= 1:
                # ヘッダーのみ or データなし
                return pd.DataFrame()

            # DataFrameに変換（1行目をヘッダーとして使用）
            df = pd.DataFrame(all_values[1:], columns=all_values[0])

            # 指定日のデータをフィルタリング
            df_filtered = df[df['日付'] == date].copy()

            # 数値列を変換
            for col in ['実測NASA_F', '予測NASA_F', '誤差(MAE)']:
                if col in df_filtered.columns:
                    df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')

            logger.info(f"Hourly Log取得: {user_id}, {date}, {len(df_filtered)}件")
            return df_filtered

        except Exception as e:
            logger.error(f"Hourly Log取得エラー: {e}")
            return pd.DataFrame()

    def save_hourly_log(self, user_id: str, hourly_data: Dict) -> bool:
        """
        時刻ごとの詳細データをスプレッドシートに保存

        シート構成: {user_id}_Hourly_Log
        列: 日付, 時刻, 活動名, 実測NASA_F, 予測NASA_F, 誤差(MAE)

        Args:
            user_id: ユーザーID
            hourly_data: {
                'date': '2025-01-15',
                'time': '10:00',
                'activity': '授業',
                'actual_frustration': 15,
                'predicted_frustration': 14.2
            }
        """
        try:
            if not self.gc:
                logger.warning("Google Sheetsクライアントが初期化されていません")
                return False

            # ユーザー別のHourly Logシート名を生成
            sheet_name = f"{user_id}_Hourly_Log"
            worksheet = self._find_worksheet_by_exact_name(sheet_name)

            if not worksheet:
                # シートが存在しない場合は作成
                worksheet = self.spreadsheet.add_worksheet(
                    title=sheet_name,
                    rows="10000",
                    cols="6"
                )
                # ヘッダー行を追加
                headers = ['日付', '時刻', '活動名', '実測NASA_F', '予測NASA_F', '誤差(MAE)']
                worksheet.update('A1:F1', [headers])
                logger.info(f"Hourly Logシートを作成しました: {sheet_name}")

            # 活動名のバリデーション
            activity = hourly_data.get('activity', '').strip()
            if not activity or activity == 'unknown':
                logger.warning(f"活動名が不正です: '{activity}' - 保存をスキップします")
                return False

            date = hourly_data.get('date', '')
            time = hourly_data.get('time', '')

            # 重複チェック: 同じ日付・時刻の行を探す
            all_values = worksheet.get_all_values()

            for idx, row in enumerate(all_values[1:], start=2):  # ヘッダーをスキップ、行番号は2から
                if len(row) >= 3 and row[0] == date and row[1] == time:
                    logger.info(f"重複を検出: {date} {time} (行{idx}) - スキップします（再予測しない）")
                    return True  # 既に存在する場合は何もしない

            # 誤差を計算
            actual = hourly_data.get('actual_frustration')
            predicted = hourly_data.get('predicted_frustration')
            mae = abs(actual - predicted) if actual is not None and predicted is not None else None

            # データ行を準備
            row_data = [
                date,
                time,
                activity,
                round(actual, 2) if actual is not None else '',
                round(predicted, 2) if predicted is not None else '',
                round(mae, 2) if mae is not None else ''
            ]

            # 新規行を追加（重複がない場合のみ）
            worksheet.append_row(row_data)
            logger.info(f"Hourly Log追加: {user_id}, {date} {time} {activity}, 実測={actual}, 予測={predicted}")

            return True

        except Exception as e:
            logger.error(f"Hourly Log保存エラー: {e}")
            return False

    def save_daily_feedback_summary(self, user_id: str, summary_data: Dict) -> bool:
        """
        日次フィードバックサマリーをスプレッドシートに保存

        シート構成: {user_id}_Daily_Summary
        列: 日付, 日次平均実測, 日次平均予測, 日次MAE, DiCE改善ポテンシャル, DiCE提案数,
            ChatGPTフィードバック, アクションプラン, 生成日時

        Args:
            user_id: ユーザーID
            summary_data: {
                'date': '2025-01-15',
                'avg_actual': 12.3,
                'avg_predicted': 11.8,
                'dice_improvement': 5.2,
                'dice_count': 3,
                'chatgpt_feedback': 'フィードバック文章',
                'action_plan': ['アクション1', 'アクション2'],
                'generated_at': '2025-01-15 21:00:00'
            }
        """
        try:
            if not self.gc:
                logger.warning("Google Sheetsクライアントが初期化されていません")
                return False

            # ユーザー別のDaily Summaryシート名を生成
            sheet_name = f"{user_id}_Daily_Summary"
            worksheet = self._find_worksheet_by_exact_name(sheet_name)

            if not worksheet:
                # シートが存在しない場合は作成
                worksheet = self.spreadsheet.add_worksheet(
                    title=sheet_name,
                    rows="1000",
                    cols="9"
                )
                # ヘッダー行を追加
                headers = [
                    '日付', '日次平均実測', '日次平均予測', '日次MAE',
                    'DiCE改善ポテンシャル', 'DiCE提案数',
                    'ChatGPTフィードバック', 'アクションプラン', '生成日時'
                ]
                worksheet.update('A1:I1', [headers])
                logger.info(f"Daily Summaryシートを作成しました: {sheet_name}")

            # 日次MAEを計算
            avg_actual = summary_data.get('avg_actual')
            avg_predicted = summary_data.get('avg_predicted')
            daily_mae = abs(avg_actual - avg_predicted) if avg_actual is not None and avg_predicted is not None else None

            # アクションプランをJSON文字列に変換
            action_plan = summary_data.get('action_plan', [])
            action_plan_str = json.dumps(action_plan, ensure_ascii=False) if action_plan else ''

            # データ行を準備
            row_data = [
                summary_data.get('date', ''),
                round(avg_actual, 2) if avg_actual is not None else '',
                round(avg_predicted, 2) if avg_predicted is not None else '',
                round(daily_mae, 2) if daily_mae is not None else '',
                round(summary_data.get('dice_improvement', 0), 2),
                summary_data.get('dice_count', 0),
                summary_data.get('chatgpt_feedback', ''),
                action_plan_str,
                summary_data.get('generated_at', datetime.now().isoformat())
            ]

            # 重複チェック: 同じ日付のデータが既に存在するかチェック
            date = summary_data.get('date', '')
            try:
                existing_data = worksheet.get_all_values()
                existing_row_index = None

                # ヘッダー行をスキップして検索
                for i, row in enumerate(existing_data[1:], start=2):
                    if len(row) > 0 and row[0] == date:
                        existing_row_index = i
                        logger.info(f"既存の日次サマリーを発見: {date} (行: {i})")
                        break

                if existing_row_index:
                    # 既存行を更新
                    cell_range = f'A{existing_row_index}:I{existing_row_index}'
                    worksheet.update(cell_range, [row_data])
                    logger.info(f"日次フィードバックサマリーを更新: {user_id}, {date}")
                else:
                    # 新規行を追加
                    worksheet.append_row(row_data)
                    logger.info(f"日次フィードバックサマリーを保存: {user_id}, {date}")

            except Exception as duplicate_check_error:
                # 重複チェックに失敗した場合は通常の追加処理を実行
                logger.warning(f"重複チェックに失敗、新規追加します: {duplicate_check_error}")
                worksheet.append_row(row_data)
                logger.info(f"日次フィードバックサマリーを保存: {user_id}, {date}")

            return True

        except Exception as e:
            logger.error(f"日次フィードバックサマリー保存エラー: {e}")
            return False