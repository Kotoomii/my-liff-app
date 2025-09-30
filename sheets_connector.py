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
    
    def get_activity_data(self, user_id: str = "default") -> pd.DataFrame:
        """
        活動データを取得（デバッグモード: Excelファイル、プロダクション: Google Sheets）
        """
        try:
            # デバッグモード: ローカルExcelファイルから読み込み
            if self.debug_mode:
                return self._get_activity_data_from_excel(user_id)
            
            # プロダクションモード: Google Sheetsから読み込み
            return self._get_activity_data_from_sheets(user_id)
            
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
    
    def get_fitbit_data(self, user_id: str = "default") -> pd.DataFrame:
        """
        生体データ（Fitbitデータ）を取得（デバッグモード: Excelファイル、プロダクション: Google Sheets）
        """
        try:
            # デバッグモード: ローカルExcelファイルから読み込み
            if self.debug_mode:
                return self._get_fitbit_data_from_excel(user_id)
            
            # プロダクションモード: Google Sheetsから読み込み
            return self._get_fitbit_data_from_sheets(user_id)
            
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