"""
アプリケーション設定ファイル
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Googleスプレッドシート設定
    SPREADSHEET_ID = os.environ.get('SPREADSHEET_ID', '15Mv9-N1RFKLmDd2vqYBLwKR-aPWA3Mgw9dAHHOWilLs')
    
    # デフォルトシート名（後方互換性のため）
    ACTIVITY_SHEET_PATTERN = "Ua06e990fd6d5f4646615595d4e8d33"  # デフォルト活動データシート名
    FITBIT_SHEET_PATTERN = "kotoomi_Fitbit-data-kotomi"  # デフォルト生体データシート名
    FIXED_PLANS_SHEET = "FIXED_PLANS"  # 固定予定シート名
    WORKLOAD_DATA_SHEET = "WORKLOAD_DATA"  # 負荷データ保存シート名
    
    # 複数ユーザー対応設定
    USER_CONFIGURATIONS = {
        'default': {
            'user_id': 'default',
            'name': 'デフォルトユーザー',
            'icon': '👤',
            'activity_sheet': 'Ua06e990fd6d5f4646615595d4e8d337f',  # LINEユーザーID
            'fitbit_sheet': 'kotoomi_Fitbit-data-kotomi',  # username_Fitbit-data-identifier
            'description': 'メインユーザー（こときみ）'
        },
        'user1': {
            'user_id': 'user1',
            'name': 'ユーザー1',
            'icon': '👨',
            'activity_sheet': 'U1234567890abcdef12345',  # LINEユーザーID例
            'fitbit_sheet': 'taro_Fitbit-data-main',  # username_Fitbit-data-identifier
            'description': 'テストユーザー1（太郎）'
        },
        'user2': {
            'user_id': 'user2', 
            'name': 'ユーザー2',
            'icon': '👩',
            'activity_sheet': 'U2345678901bcdefg23456',  # LINEユーザーID例
            'fitbit_sheet': 'hanako_Fitbit-data-main',  # username_Fitbit-data-identifier
            'description': 'テストユーザー2（花子）'
        },
        'user3': {
            'user_id': 'user3',
            'name': 'ユーザー3', 
            'icon': '🧑',
            'activity_sheet': 'U3456789012cdefgh34567',  # LINEユーザーID例
            'fitbit_sheet': 'jiro_Fitbit-data-secondary',  # username_Fitbit-data-identifier
            'description': 'テストユーザー3（次郎）'
        }
    }
    
    @classmethod
    def get_user_config(cls, user_id: str = 'default'):
        """指定ユーザーの設定を取得"""
        return cls.USER_CONFIGURATIONS.get(user_id, cls.USER_CONFIGURATIONS['default'])
    
    @classmethod
    def get_activity_sheet_name(cls, user_id: str = 'default'):
        """指定ユーザーの活動データシート名を取得"""
        user_config = cls.get_user_config(user_id)
        return user_config['activity_sheet']
    
    @classmethod
    def get_fitbit_sheet_name(cls, user_id: str = 'default'):
        """指定ユーザーのFitbitデータシート名を取得"""
        user_config = cls.get_user_config(user_id)
        return user_config['fitbit_sheet']
    
    @classmethod
    def get_username_from_fitbit_sheet(cls, user_id: str = 'default'):
        """Fitbitシートからユーザーネームを抽出（アンダースコアの前）"""
        fitbit_sheet = cls.get_fitbit_sheet_name(user_id)
        return fitbit_sheet.split('_')[0] if '_' in fitbit_sheet else fitbit_sheet
    
    @classmethod
    def get_all_users(cls):
        """全ユーザー設定を取得"""
        return list(cls.USER_CONFIGURATIONS.values())
    
    # 機械学習モデル設定
    MODEL_TYPE = 'RandomForest'
    N_ESTIMATORS = 100
    MAX_DEPTH = 10
    RANDOM_STATE = 42
    
    # DiCE設定
    COUNTERFACTUAL_COUNT = 3
    DESIRED_STRESS_RANGE = [0, 40]
    
    # NASA-TLX項目
    NASA_DIMENSIONS = ['NASA_M', 'NASA_P', 'NASA_T', 'NASA_O', 'NASA_E', 'NASA_F']
    NASA_LABELS = {
        'NASA_M': '精神的要求',
        'NASA_P': '身体的要求', 
        'NASA_T': '時間的切迫感',
        'NASA_O': '達成度',
        'NASA_E': '努力',
        'NASA_F': 'フラストレーション'
    }
    
    # ログレベル
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
    # デバッグモード
    DEBUG = os.environ.get('FLASK_ENV') == 'development'