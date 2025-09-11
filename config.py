"""
アプリケーション設定ファイル
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Googleスプレッドシート設定
    SPREADSHEET_ID = os.environ.get('SPREADSHEET_ID', '15Mv9-N1RFKLmDd2vqYBLwKR-aPWA3Mgw9dAHHOWilLs')
    
    # データシート名設定
    ACTIVITY_SHEET_PATTERN = "Ua06e990fd6d5f4646615595d4e8d33"  # 活動データのシート名パターン
    FITBIT_SHEET_PATTERN = "kotoomi_Fitbit-data-kotomi"  # 生体データのシート名パターン
    FIXED_PLANS_SHEET = "FIXED_PLANS"  # 固定予定シート名
    WORKLOAD_DATA_SHEET = "WORKLOAD_DATA"  # 負荷データ保存シート名
    
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