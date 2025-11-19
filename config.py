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
    ACTIVITY_SHEET_PATTERN = "Ua06e990fd6d5f4646615595d4e8d337f"  # デフォルト活動データシート名
    FITBIT_SHEET_PATTERN = "kotoomi_Fitbit-data-kotomi"  # デフォルト生体データシート名
    FIXED_PLANS_SHEET = "FIXED_PLANS"  # 固定予定シート名
    WORKLOAD_DATA_SHEET = "WORKLOAD_DATA"  # 負荷データ保存シート名
    
    # 複数ユーザー対応設定
    USER_CONFIGURATIONS = {
        'default': {
            'user_id': 'default',
            'name': 'デフォルトユーザー',
            'icon': '👤',
            'activity_sheet': 'Ua06e990fd6d5f4646615595d4e8d337f',  # LINEユーザーID (Excel対応)
            'fitbit_sheet': 'kotoomi_Fitbit-data-kotomi',  # username_Fitbit-data-identifier
            'description': 'メインユーザー（こときみ）'
        },
        'user1': {
            'user_id': 'user1',
            'name': '小手川',
            'icon': '👨',
            'activity_sheet': 'U1234567890abcdef12345',  # LINEユーザーID例
            'fitbit_sheet': 'taro_Fitbit-data-main',  # username_Fitbit-data-identifier
            'description': '小手川さん'
        },
        'user2': {
            'user_id': 'user2',
            'name': '榎本',
            'icon': '👩',
            'activity_sheet': 'U2345678901bcdefg23456',  # LINEユーザーID例
            'fitbit_sheet': 'hanako_Fitbit-data-main',  # username_Fitbit-data-identifier
            'description': '榎本さん'
        },
        'user3': {
            'user_id': 'user3',
            'name': '長山',
            'icon': '🧑',
            'activity_sheet': 'U3456789012cdefgh34567',  # LINEユーザーID例
            'fitbit_sheet': 'jiro_Fitbit-data-secondary',  # username_Fitbit-data-identifier
            'description': '長山さん'
        },
        'user4': {
            'user_id': 'user4',
            'name': '柴田',
            'icon': '👦',
            'activity_sheet': 'U4567890123defghi45678',  # LINEユーザーID例
            'fitbit_sheet': 'user4_Fitbit-data-main',
            'description': '柴田さん'
        },
        'user5': {
            'user_id': 'user5',
            'name': '竹田',
            'icon': '👧',
            'activity_sheet': 'U5678901234efghij56789',  # LINEユーザーID例
            'fitbit_sheet': 'user5_Fitbit-data-main',
            'description': '竹田さん'
        },
        'user6': {
            'user_id': 'user6',
            'name': '新名',
            'icon': '🧒',
            'activity_sheet': 'U6789012345fghijk67890',  # LINEユーザーID例
            'fitbit_sheet': 'user6_Fitbit-data-main',
            'description': '新名さん'
        },
        'user7': {
            'user_id': 'user7',
            'name': '寺岡',
            'icon': '👨‍🦱',
            'activity_sheet': 'U7890123456ghijkl78901',  # LINEユーザーID例
            'fitbit_sheet': 'user7_Fitbit-data-main',
            'description': '寺岡さん'
        },
        'user8': {
            'user_id': 'user8',
            'name': '前地',
            'icon': '👩‍🦱',
            'activity_sheet': 'U8901234567hijklm89012',  # LINEユーザーID例
            'fitbit_sheet': 'user8_Fitbit-data-main',
            'description': '前地さん'
        },
        'user9': {
            'user_id': 'user9',
            'name': 'ユーザー9',
            'icon': '🧑‍🦱',
            'activity_sheet': 'U9012345678ijklmn90123',  # LINEユーザーID例
            'fitbit_sheet': 'user9_Fitbit-data-main',
            'description': 'テストユーザー9'
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
    # MODEL_TYPE: 'RandomForest', 'Linear', 'SVR' を指定
    MODEL_TYPE = 'SVR'  # 'RandomForest', 'Linear', 'SVR' から選択

    # RandomForest用パラメータ
    N_ESTIMATORS = 100
    MAX_DEPTH = 10
    RANDOM_STATE = 42

    # SVR用パラメータ
    SVR_KERNEL = 'rbf'  # 'linear', 'poly', 'rbf', 'sigmoid'
    SVR_C = 1.0  # 正則化パラメータ（大きいほど過学習しやすい）
    SVR_EPSILON = 0.1  # イプシロンチューブの幅
    SVR_GAMMA = 'scale'  # rbfカーネルの係数（'scale'または'auto'）
    
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
    
    # ログレベル設定
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'WARNING')  # Cloud Run本番環境ではWARNING推奨

    # デバッグモード
    DEBUG = os.environ.get('FLASK_ENV') == 'development'

    # Cloud Run環境判定
    IS_CLOUD_RUN = os.environ.get('K_SERVICE') is not None

    # ログ出力設定
    ENABLE_DEBUG_LOGS = os.environ.get('ENABLE_DEBUG_LOGS', 'false').lower() == 'true'
    ENABLE_INFO_LOGS = os.environ.get('ENABLE_INFO_LOGS', 'false').lower() == 'true'

    # 詳細ログ出力を制御するフラグ
    LOG_PREDICTIONS = os.environ.get('LOG_PREDICTIONS', 'false').lower() == 'true'
    LOG_DATA_OPERATIONS = os.environ.get('LOG_DATA_OPERATIONS', 'false').lower() == 'true'
    LOG_MODEL_TRAINING = os.environ.get('LOG_MODEL_TRAINING', 'false').lower() == 'true'