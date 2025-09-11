"""
行動単位の反実仮想説明（Counterfactual Explanations）機能
DiCEライブラリを使用してフラストレーション値改善提案を生成
過去24時間の行動をもとに、どのように行動を変えていたらフラストレーション値が下がっていたかを示す
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import dice_ml
from datetime import datetime, timedelta
import json

from config import Config

logger = logging.getLogger(__name__)

class ActivityCounterfactualExplainer:
    def __init__(self):
        self.config = Config()
        self.dice_exp = None
        self.predictor = None
        
    def generate_activity_based_explanation(self, 
                                          df_enhanced: pd.DataFrame, 
                                          predictor,
                                          target_timestamp: datetime = None,
                                          lookback_hours: int = 24) -> Dict:
        """
        行動単位の反実仮想説明を生成
        過去24時間の行動をもとに、どのように変更すればフラストレーション値が下がったかを示す
        """
        try:
            self.predictor = predictor
            
            if target_timestamp is None:
                target_timestamp = datetime.now()
            
            # 過去24時間の行動変更タイミングを取得
            cutoff_time = target_timestamp - timedelta(hours=lookback_hours)
            recent_activities = df_enhanced[
                (df_enhanced['Timestamp'] >= cutoff_time) & 
                (df_enhanced['Timestamp'] <= target_timestamp) &
                (df_enhanced['activity_change'] == 1)
            ].copy()
            
            if recent_activities.empty:
                return self._get_fallback_explanation()
            
            # 各行動変更タイミングについて反実仮想例を生成
            counterfactual_results = []
            
            for idx, activity in recent_activities.iterrows():
                cf_result = self._generate_single_activity_counterfactual(
                    df_enhanced, idx, activity
                )
                if cf_result:
                    counterfactual_results.append(cf_result)
            
            if not counterfactual_results:
                return self._get_fallback_explanation()
            
            # 結果をまとめて返す
            return self._summarize_counterfactual_results(counterfactual_results)
            
        except Exception as e:
            logger.error(f"行動単位反実仮想説明生成エラー: {e}")
            return self._get_fallback_explanation()
    
    def _generate_single_activity_counterfactual(self, 
                                               df_enhanced: pd.DataFrame, 
                                               activity_idx: int,
                                               activity: pd.Series) -> Optional[Dict]:
        """
        単一の行動について反実仮想例を生成
        """
        try:
            # 現在の特徴量を取得
            original_idx = df_enhanced.index.get_loc(activity_idx)
            features = self.predictor.create_features_for_activity(df_enhanced, original_idx)
            
            if features is None:
                return None
            
            # 現在のフラストレーション値を予測
            current_prediction = self.predictor.predict_frustration_at_activity_change(
                df_enhanced, activity['Timestamp']
            )
            
            current_frustration = current_prediction.get('predicted_frustration', activity.get('NASA_F', 10))
            
            # 異なる活動パターンでの予測を計算
            alternative_activities = self._get_alternative_activities(activity)
            counterfactuals = []
            
            for alt_activity in alternative_activities:
                # 特徴量を変更して予測
                modified_features = features.copy()
                
                # 活動カテゴリを変更
                if 'current_activity' in self.predictor.encoders:
                    try:
                        alt_encoded = self.predictor.encoders['current_activity'].transform([alt_activity])[0]
                        modified_features['current_activity'] = alt_encoded
                    except ValueError:
                        continue  # 未知の活動はスキップ
                
                # 活動の種類に応じて期間を調整
                modified_features['current_duration'] = self._get_typical_duration(alt_activity)
                
                # 予測実行
                pred_df = pd.DataFrame([modified_features])
                
                # 不足している特徴量列を0で埋める
                for col in self.predictor.feature_columns:
                    if col not in pred_df.columns:
                        pred_df[col] = 0.0
                
                pred_df = pred_df[self.predictor.feature_columns]
                
                if self.predictor.model is not None:
                    alt_prediction = self.predictor.model.predict(pred_df)[0]
                    
                    # フラストレーション値が改善される場合のみ記録
                    if alt_prediction < current_frustration:
                        improvement = current_frustration - alt_prediction
                        counterfactuals.append({
                            'original_activity': activity.get('CatSub', 'unknown'),
                            'alternative_activity': alt_activity,
                            'original_frustration': current_frustration,
                            'alternative_frustration': alt_prediction,
                            'improvement': improvement,
                            'timestamp': activity['Timestamp'],
                            'duration': activity.get('Duration', 0),
                            'confidence': min(0.9, 0.6 + 0.3 * (improvement / 30))
                        })
            
            # 最も改善効果の高い代替案を選択
            if counterfactuals:
                best_cf = max(counterfactuals, key=lambda x: x['improvement'])
                return best_cf
            
            return None
            
        except Exception as e:
            logger.error(f"単一活動反実仮想例生成エラー: {e}")
            return None
    
    def _get_alternative_activities(self, activity: pd.Series) -> List[str]:
        """
        現在の活動に対する代替活動候補を取得
        """
        current_activity = activity.get('CatSub', '')
        hour = activity.get('hour', 12)
        is_weekend = activity.get('is_weekend', 0)
        
        # 時間帯と曜日に応じた代替活動を提案
        alternatives = []
        
        # 基本的な低ストレス活動
        low_stress_activities = ['休憩', '軽い運動', '散歩', '音楽鑑賞', '読書']
        
        # 時間帯別の適切な活動
        if 5 <= hour < 9:  # 朝
            alternatives.extend(['朝食', '軽い運動', '瞑想', '読書'])
        elif 9 <= hour < 12:  # 午前
            if is_weekend:
                alternatives.extend(['趣味', '散歩', '家事', '読書'])
            else:
                alternatives.extend(['軽い仕事', '会議', '資料作成'])
        elif 12 <= hour < 14:  # 昼
            alternatives.extend(['昼食', '昼休み', '軽い散歩'])
        elif 14 <= hour < 18:  # 午後
            if is_weekend:
                alternatives.extend(['趣味', '運動', '友人との時間'])
            else:
                alternatives.extend(['軽い仕事', '会議', '資料整理'])
        elif 18 <= hour < 21:  # 夕方〜夜
            alternatives.extend(['夕食', '家族時間', '趣味', 'テレビ'])
        else:  # 夜〜深夜
            alternatives.extend(['入浴', '読書', '音楽鑑賞', '睡眠準備'])
        
        # 現在の活動が高ストレスの場合は低ストレス活動を多く含める
        if current_activity in ['仕事', '会議', '勉強', '資料作成']:
            alternatives.extend(low_stress_activities)
        
        # 現在の活動を除外して重複を削除
        alternatives = list(set(alternatives))
        if current_activity in alternatives:
            alternatives.remove(current_activity)
        
        return alternatives[:5]  # 最大5つの代替案
    
    def _get_typical_duration(self, activity: str) -> float:
        """
        活動タイプに応じた典型的な所要時間を返す（分）
        """
        duration_map = {
            '食事': 45, '朝食': 30, '昼食': 45, '夕食': 60,
            '仕事': 120, '会議': 60, '資料作成': 90,
            '運動': 60, '軽い運動': 30, '散歩': 30,
            '休憩': 15, '昼休み': 30,
            '読書': 45, '音楽鑑賞': 30, 'テレビ': 60,
            '入浴': 30, '睡眠準備': 30,
            '趣味': 90, '家事': 45, '友人との時間': 120,
            '瞑想': 20, '資料整理': 45
        }
        
        return duration_map.get(activity, 60)  # デフォルト60分
    
    def _summarize_counterfactual_results(self, counterfactual_results: List[Dict]) -> Dict:
        """
        反実仮想結果をまとめて説明を生成
        """
        try:
            if not counterfactual_results:
                return self._get_fallback_explanation()
            
            # 改善効果順にソート
            counterfactual_results.sort(key=lambda x: x['improvement'], reverse=True)
            
            total_improvement = sum(cf['improvement'] for cf in counterfactual_results)
            avg_improvement = total_improvement / len(counterfactual_results)
            
            # 時間軸での結果を整理
            timeline_results = []
            for cf in counterfactual_results:
                timeline_results.append({
                    'timestamp': cf['timestamp'],
                    'hour': cf['timestamp'].hour,
                    'original_activity': cf['original_activity'],
                    'suggested_activity': cf['alternative_activity'],
                    'frustration_reduction': cf['improvement'],
                    'original_frustration': cf['original_frustration'],
                    'predicted_frustration': cf['alternative_frustration']
                })
            
            # 時間順にソート
            timeline_results.sort(key=lambda x: x['timestamp'])
            
            # 主要な提案をまとめる
            top_suggestions = counterfactual_results[:3]  # 上位3つ
            
            suggestions = []
            for cf in top_suggestions:
                suggestion = f"{cf['timestamp'].strftime('%H:%M')} - " \
                           f"「{cf['original_activity']}」を「{cf['alternative_activity']}」に変更すると" \
                           f"フラストレーション値が{cf['improvement']:.1f}点改善"
                suggestions.append(suggestion)
            
            # 活動カテゴリ別の分析
            activity_analysis = self._analyze_activity_patterns(counterfactual_results)
            
            return {
                'type': 'activity_counterfactual',
                'total_improvement': total_improvement,
                'average_improvement': avg_improvement,
                'num_suggestions': len(counterfactual_results),
                'timeline': timeline_results,
                'top_suggestions': suggestions,
                'activity_analysis': activity_analysis,
                'confidence': min(0.9, 0.6 + 0.3 * len(counterfactual_results) / 10),
                'summary': f"過去24時間で{len(counterfactual_results)}個の行動変更により、"
                          f"平均{avg_improvement:.1f}点のフラストレーション改善が期待できます。"
            }
            
        except Exception as e:
            logger.error(f"反実仮想結果まとめエラー: {e}")
            return self._get_fallback_explanation()
    
    def _analyze_activity_patterns(self, counterfactual_results: List[Dict]) -> Dict:
        """
        活動パターンを分析して傾向を把握
        """
        try:
            # 改善提案の多い活動を特定
            original_activities = {}
            suggested_activities = {}
            
            for cf in counterfactual_results:
                orig = cf['original_activity']
                sugg = cf['alternative_activity']
                
                if orig not in original_activities:
                    original_activities[orig] = {'count': 0, 'total_improvement': 0}
                original_activities[orig]['count'] += 1
                original_activities[orig]['total_improvement'] += cf['improvement']
                
                if sugg not in suggested_activities:
                    suggested_activities[sugg] = {'count': 0, 'total_improvement': 0}
                suggested_activities[sugg]['count'] += 1
                suggested_activities[sugg]['total_improvement'] += cf['improvement']
            
            # 最も問題のある活動
            problematic_activity = max(original_activities.items(), 
                                     key=lambda x: x[1]['total_improvement'])
            
            # 最も推奨される活動
            recommended_activity = max(suggested_activities.items(),
                                     key=lambda x: x[1]['total_improvement'])
            
            # 時間帯分析
            time_analysis = {}
            for cf in counterfactual_results:
                hour = cf['timestamp'].hour
                time_period = self._get_time_period(hour)
                
                if time_period not in time_analysis:
                    time_analysis[time_period] = {'count': 0, 'improvement': 0}
                time_analysis[time_period]['count'] += 1
                time_analysis[time_period]['improvement'] += cf['improvement']
            
            return {
                'most_problematic_activity': {
                    'activity': problematic_activity[0],
                    'frequency': problematic_activity[1]['count'],
                    'total_improvement_potential': problematic_activity[1]['total_improvement']
                },
                'most_recommended_activity': {
                    'activity': recommended_activity[0],
                    'frequency': recommended_activity[1]['count'],
                    'total_benefit': recommended_activity[1]['total_improvement']
                },
                'time_period_analysis': time_analysis
            }
            
        except Exception as e:
            logger.error(f"活動パターン分析エラー: {e}")
            return {}
    
    def _get_time_period(self, hour: int) -> str:
        """時間を時間帯に変換"""
        if 5 <= hour < 12:
            return '朝'
        elif 12 <= hour < 18:
            return '午後'
        elif 18 <= hour < 22:
            return '夕方'
        else:
            return '夜'
    
    def generate_daily_summary(self, 
                             past_24h_explanations: List[Dict],
                             target_date: str = None) -> Dict:
        """
        過去24時間のDiCE結果をまとめて1日の総合的なフィードバックを生成
        """
        try:
            if target_date is None:
                target_date = datetime.now().strftime('%Y-%m-%d')
            
            if not past_24h_explanations:
                return self._get_fallback_daily_summary()
            
            # 全ての提案を統合
            all_suggestions = []
            total_improvement = 0
            activity_counts = {}
            time_period_improvements = {'朝': 0, '午後': 0, '夕方': 0, '夜': 0}
            
            for explanation in past_24h_explanations:
                if explanation.get('type') == 'activity_counterfactual':
                    total_improvement += explanation.get('total_improvement', 0)
                    
                    for timeline_item in explanation.get('timeline', []):
                        all_suggestions.append(timeline_item)
                        
                        # 活動カウント
                        orig_activity = timeline_item['original_activity']
                        activity_counts[orig_activity] = activity_counts.get(orig_activity, 0) + 1
                        
                        # 時間帯別改善
                        time_period = self._get_time_period(timeline_item['hour'])
                        time_period_improvements[time_period] += timeline_item['frustration_reduction']
            
            # 最も改善効果の高い時間帯
            best_time_period = max(time_period_improvements.items(), key=lambda x: x[1])
            
            # 最も頻繁に問題となる活動
            most_frequent_problem = max(activity_counts.items(), key=lambda x: x[1]) if activity_counts else ('不明', 0)
            
            # 全体的なアドバイス生成
            overall_advice = self._generate_overall_advice(
                total_improvement, 
                best_time_period, 
                most_frequent_problem,
                len(all_suggestions)
            )
            
            return {
                'date': target_date,
                'type': 'daily_summary',
                'total_potential_improvement': total_improvement,
                'number_of_improvement_opportunities': len(all_suggestions),
                'best_improvement_time_period': {
                    'period': best_time_period[0],
                    'potential_improvement': best_time_period[1]
                },
                'most_frequent_problematic_activity': {
                    'activity': most_frequent_problem[0],
                    'frequency': most_frequent_problem[1]
                },
                'time_period_breakdown': time_period_improvements,
                'overall_advice': overall_advice,
                'detailed_timeline': sorted(all_suggestions, key=lambda x: x['timestamp'])[:10]  # 上位10件
            }
            
        except Exception as e:
            logger.error(f"日次まとめ生成エラー: {e}")
            return self._get_fallback_daily_summary()
    
    def _generate_overall_advice(self, 
                               total_improvement: float,
                               best_time_period: Tuple[str, float],
                               most_frequent_problem: Tuple[str, int],
                               num_opportunities: int) -> List[str]:
        """
        総合的なアドバイスを生成
        """
        advice = []
        
        if total_improvement > 50:
            advice.append(f"今日は{total_improvement:.1f}点のフラストレーション改善の機会がありました。")
        elif total_improvement > 20:
            advice.append(f"{total_improvement:.1f}点の改善機会がありました。小さな変更でもストレス軽減につながります。")
        else:
            advice.append("全体的に良いストレス管理ができていました。この調子で続けましょう。")
        
        if best_time_period[1] > 15:
            advice.append(f"{best_time_period[0]}の時間帯に最も改善の余地があります（{best_time_period[1]:.1f}点）。")
        
        if most_frequent_problem[1] > 2:
            activity_advice = {
                '仕事': '仕事の合間により多くの休憩を取ることを検討してください。',
                '会議': '会議の前後にリラックスタイムを設けることが効果的です。',
                '勉強': '勉強時間を細切れにして、集中力を維持しましょう。',
                '家事': '家事を楽しい音楽と一緒に行うことでストレスを軽減できます。'
            }
            
            if most_frequent_problem[0] in activity_advice:
                advice.append(activity_advice[most_frequent_problem[0]])
            else:
                advice.append(f"{most_frequent_problem[0]}の時間をより快適に過ごす方法を探してみましょう。")
        
        # 一般的なアドバイス
        if num_opportunities > 5:
            advice.append("多くの改善機会がありました。小さな変更から始めて、徐々に生活パターンを調整しましょう。")
        
        advice.append("明日はこれらの提案を参考に、より快適な一日を過ごしてください。")
        
        return advice
    
    def _get_fallback_explanation(self) -> Dict:
        """フォールバック用の基本的な説明"""
        return {
            'type': 'fallback',
            'total_improvement': 0,
            'timeline': [],
            'suggestions': [
                "定期的な休憩を取り、深呼吸をしましょう",
                "軽い運動や散歩でリフレッシュしてください",
                "好きな音楽を聴いてリラックスしましょう",
                "十分な睡眠時間を確保してください"
            ],
            'confidence': 0.3
        }
    
    def _get_fallback_daily_summary(self) -> Dict:
        """フォールバック用の日次サマリー"""
        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'type': 'fallback_daily_summary',
            'total_potential_improvement': 0,
            'overall_advice': [
                "規則正しい生活リズムを心がけましょう",
                "ストレスを感じたら深呼吸をしてリラックスしましょう",
                "適度な運動と良質な睡眠を大切にしてください",
                "明日も健康的な一日を過ごしましょう"
            ],
            'confidence': 0.3
        }