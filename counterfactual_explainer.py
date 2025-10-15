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

            if self.config.ENABLE_DEBUG_LOGS:
                logger.debug(f"DiCE説明生成開始: target_timestamp={target_timestamp}, lookback_hours={lookback_hours}")

            # データ存在チェック
            if df_enhanced.empty:
                logger.warning("DiCE説明生成: データが空です")
                return self._get_no_data_explanation("データが空です")

            # 過去24時間の行動変更タイミングを取得
            cutoff_time = target_timestamp - timedelta(hours=lookback_hours)
            recent_activities = df_enhanced[
                (df_enhanced['Timestamp'] >= cutoff_time) &
                (df_enhanced['Timestamp'] <= target_timestamp) &
                (df_enhanced['activity_change'] == 1)
            ].copy()

            if self.config.ENABLE_DEBUG_LOGS:
                logger.debug(f"対象期間の活動数: {len(recent_activities)}")

            if recent_activities.empty:
                logger.info(f"DiCE説明生成: 指定期間（{cutoff_time}〜{target_timestamp}）に行動変更タイミングがありません")
                return self._get_no_solution_explanation(
                    f"指定期間に行動変更タイミングがありませんでした",
                    details={
                        'period_start': cutoff_time.isoformat(),
                        'period_end': target_timestamp.isoformat(),
                        'total_activities_in_period': len(df_enhanced[
                            (df_enhanced['Timestamp'] >= cutoff_time) &
                            (df_enhanced['Timestamp'] <= target_timestamp)
                        ]),
                        'reason': 'no_activity_changes'
                    }
                )

            # 各行動変更タイミングについて反実仮想例を生成
            counterfactual_results = []
            failed_generations = 0

            for idx, activity in recent_activities.iterrows():
                try:
                    cf_result = self._generate_single_activity_counterfactual(
                        df_enhanced, idx, activity
                    )
                    if cf_result:
                        counterfactual_results.append(cf_result)
                    else:
                        failed_generations += 1
                        if self.config.ENABLE_DEBUG_LOGS:
                            logger.debug(f"活動 {activity.get('CatSub', 'unknown')} の代替案生成に失敗")
                except Exception as cf_error:
                    failed_generations += 1
                    logger.warning(f"活動 {activity.get('CatSub', 'unknown')} の反実仮想例生成中にエラー: {cf_error}")

            logger.info(f"DiCE生成結果: 成功={len(counterfactual_results)}, 失敗={failed_generations}, 合計={len(recent_activities)}")

            if not counterfactual_results:
                logger.warning(f"DiCE説明生成: すべての活動で代替案を生成できませんでした（対象活動数: {len(recent_activities)}）")
                return self._get_no_solution_explanation(
                    f"{len(recent_activities)}個の活動を分析しましたが、改善提案を生成できませんでした",
                    details={
                        'total_activities_analyzed': len(recent_activities),
                        'failed_generations': failed_generations,
                        'reason': 'no_improvements_found',
                        'possible_causes': [
                            'モデルの学習データが不足している',
                            '既にフラストレーション値が低く、改善の余地が少ない',
                            '代替活動の選択肢が限定されている'
                        ]
                    }
                )

            # 結果をまとめて返す
            return self._summarize_counterfactual_results(counterfactual_results)

        except Exception as e:
            logger.error(f"行動単位反実仮想説明生成で予期しないエラー: {e}", exc_info=True)
            return self._get_error_explanation(str(e))
    
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

            # 提案の多様性をチェック
            unique_activities = len(set(cf['alternative_activity'] for cf in counterfactual_results))
            is_diverse = unique_activities >= min(3, len(counterfactual_results))
            
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
            
            result = {
                'type': 'activity_counterfactual',
                'total_improvement': total_improvement,
                'average_improvement': avg_improvement,
                'num_suggestions': len(counterfactual_results),
                'timeline': timeline_results,
                'top_suggestions': suggestions,
                'activity_analysis': activity_analysis,
                'confidence': min(0.9, 0.6 + 0.3 * len(counterfactual_results) / 10),
                'summary': f"過去24時間で{len(counterfactual_results)}個の行動変更により、"
                          f"平均{avg_improvement:.1f}点のフラストレーション改善が期待できます。",
                'diversity_check': {
                    'unique_suggestions': unique_activities,
                    'total_suggestions': len(counterfactual_results),
                    'is_diverse': is_diverse
                }
            }

            # 多様性が低い場合の警告
            if not is_diverse:
                result['warning'] = {
                    'message': 'DiCEの提案が単調になっています。データの多様性が不足している可能性があります。',
                    'unique_activities': unique_activities,
                    'recommendation': '様々な種類の活動データを記録することで、より多様な提案が可能になります。'
                }

            return result
            
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
    
    def _get_error_explanation(self, error_message: str) -> Dict:
        """エラー発生時の説明"""
        return {
            'type': 'error',
            'status': 'error',
            'error_message': error_message,
            'total_improvement': 0,
            'timeline': [],
            'suggestions': [
                "定期的な休憩を取り、深呼吸をしましょう",
                "軽い運動や散歩でリフレッシュしてください",
                "好きな音楽を聴いてリラックスしましょう",
                "十分な睡眠時間を確保してください"
            ],
            'confidence': 0.0,
            'user_message': 'DiCEの実行中にエラーが発生しました。システム管理者に連絡してください。'
        }

    def _get_no_data_explanation(self, reason: str, details: dict = None) -> Dict:
        """データが不足している場合の説明"""
        result = {
            'type': 'no_data',
            'status': 'no_data',
            'reason': reason,
            'total_improvement': 0,
            'timeline': [],
            'suggestions': [
                "活動データを継続的に記録してください",
                "より多くのデータが蓄積されると、詳細な提案が可能になります",
                "定期的な休憩を取り、深呼吸をしましょう",
                "軽い運動や散歩でリフレッシュしてください"
            ],
            'confidence': 0.1,
            'user_message': 'データが不足しているため、DiCEによる改善提案を生成できませんでした。'
        }
        if details:
            result['details'] = details
        return result

    def _get_no_solution_explanation(self, reason: str, details: dict = None) -> Dict:
        """改善提案が見つからない場合の説明（データはあるが解なし）"""
        result = {
            'type': 'no_solution',
            'status': 'no_solution',
            'reason': reason,
            'total_improvement': 0,
            'timeline': [],
            'suggestions': [
                "現在のフラストレーション値は既に低い水準にあります",
                "健康的な生活リズムを維持しましょう",
                "定期的な休憩を取り、深呼吸をしましょう",
                "軽い運動や散歩でリフレッシュしてください"
            ],
            'confidence': 0.3,
            'user_message': 'データを分析しましたが、現時点では大きな改善提案を生成できませんでした。'
        }
        if details:
            result['details'] = details
        return result

    def _get_fallback_explanation(self) -> Dict:
        """フォールバック用の基本的な説明（後方互換性のため保持）"""
        return self._get_no_solution_explanation("代替案を生成できませんでした")
    
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
    
    def generate_hourly_alternatives(self, activities_data: pd.DataFrame,
                                   predictor, target_date: datetime = None) -> dict:
        """
        1日の終わりに時間単位の粒度でDiCE改善提案を生成
        """
        try:
            if target_date is None:
                target_date = datetime.now().date()

            if self.config.ENABLE_DEBUG_LOGS:
                logger.debug(f"時間別DiCE提案生成開始: target_date={target_date}")

            # データ存在チェック
            if activities_data.empty:
                logger.warning("時間別DiCE提案生成: 活動データが空です")
                return self._get_no_data_hourly_schedule("活動データが空です")

            # 指定日のデータを抽出
            day_data = activities_data[
                activities_data['Timestamp'].dt.date == target_date
            ].copy()

            if day_data.empty:
                logger.info(f"時間別DiCE提案生成: {target_date}のデータが見つかりません")
                return self._get_no_data_hourly_schedule(
                    f"{target_date}のデータが見つかりませんでした",
                    details={
                        'target_date': target_date.strftime('%Y-%m-%d'),
                        'reason': 'no_data_for_date',
                        'total_records': len(activities_data)
                    }
                )

            logger.info(f"対象日の活動データ数: {len(day_data)}")

            # 時間別の改善提案を生成
            hourly_schedule = []
            total_improvement = 0
            generation_errors = 0

            # 24時間分のスケジュールを生成
            for hour in range(24):
                hour_start = datetime.combine(target_date, datetime.min.time()) + timedelta(hours=hour)
                hour_end = hour_start + timedelta(hours=1)

                # この時間帯の活動データを取得
                hour_activities = day_data[
                    (day_data['Timestamp'] >= hour_start) &
                    (day_data['Timestamp'] < hour_end)
                ]

                if not hour_activities.empty:
                    # 実際の活動があった時間帯
                    original_activity = hour_activities.iloc[0]

                    try:
                        # 代替活動の提案を生成
                        alternative_suggestion = self._generate_hourly_alternative(
                            original_activity, predictor, hour
                        )

                        if alternative_suggestion:
                            hourly_schedule.append(alternative_suggestion)
                            total_improvement += alternative_suggestion.get('improvement', 0)
                        else:
                            if self.config.ENABLE_DEBUG_LOGS:
                                logger.debug(f"{hour}時台: 代替活動の生成に失敗（活動: {original_activity.get('CatSub', 'unknown')}）")
                    except Exception as hour_error:
                        generation_errors += 1
                        logger.warning(f"{hour}時台の代替活動生成中にエラー: {hour_error}")
                else:
                    # 活動データがない時間帯は推奨活動を提案
                    recommended_activity = self._recommend_activity_for_hour(hour)
                    hourly_schedule.append({
                        'hour': hour,
                        'time_range': f"{hour:02d}:00-{hour+1:02d}:00",
                        'original_activity': '未記録',
                        'suggested_activity': recommended_activity,
                        'improvement': 0,
                        'reason': '健康的な生活リズムのため',
                        'confidence': 0.6
                    })

            logger.info(f"時間別DiCE生成結果: 提案数={len(hourly_schedule)}, 合計改善={total_improvement:.1f}, エラー={generation_errors}")

            # 改善提案が全くない場合
            if total_improvement == 0 and generation_errors == 0:
                logger.warning(f"時間別DiCE提案生成: 改善提案を生成できませんでした（対象活動数: {len(day_data)}）")
                return self._get_no_solution_hourly_schedule(
                    f"{len(day_data)}個の活動を分析しましたが、改善提案を生成できませんでした",
                    details={
                        'target_date': target_date.strftime('%Y-%m-%d'),
                        'total_activities': len(day_data),
                        'reason': 'no_improvements_found',
                        'possible_causes': [
                            'モデルの学習データが不足している',
                            'フラストレーション値が既に低く、改善の余地が少ない',
                            '代替活動の選択肢が限定されている'
                        ]
                    }
                )

            # 改善効果の高い時間帯をハイライト
            significant_improvements = [
                item for item in hourly_schedule
                if item.get('improvement', 0) > 3
            ]

            # 提案の多様性チェック
            suggested_activities = [item.get('suggested_activity') for item in hourly_schedule if item.get('suggested_activity')]
            unique_suggestions = len(set(suggested_activities))
            is_diverse = unique_suggestions >= min(3, len(suggested_activities)) if suggested_activities else False

            result = {
                'type': 'hourly_dice_schedule',
                'date': target_date.strftime('%Y-%m-%d'),
                'hourly_schedule': hourly_schedule,
                'total_improvement': total_improvement,
                'average_improvement': total_improvement / 24 if hourly_schedule else 0,
                'significant_improvements': significant_improvements,
                'message': f"今日このような活動をしていたらストレスレベルが{total_improvement:.1f}点下がっていました",
                'confidence': min(0.9, 0.5 + len(significant_improvements) * 0.1),
                'summary': f"24時間中{len(significant_improvements)}時間で大きな改善の可能性がありました",
                'diversity_check': {
                    'unique_suggestions': unique_suggestions,
                    'total_suggestions': len(suggested_activities),
                    'is_diverse': is_diverse
                },
                'generation_stats': {
                    'total_hours_analyzed': 24,
                    'hours_with_activities': len([item for item in hourly_schedule if item.get('original_activity') != '未記録']),
                    'hours_with_improvements': len(significant_improvements),
                    'generation_errors': generation_errors
                }
            }

            # 多様性が低い場合の警告
            if not is_diverse and len(suggested_activities) > 0:
                result['warning'] = {
                    'message': 'DiCEの時間別提案が単調になっています。学習データが不足しているか、偏っている可能性があります。',
                    'unique_suggestions': unique_suggestions,
                    'recommendation': 'より多様な活動データを記録することで、個別化された提案が可能になります。'
                }

            return result

        except Exception as e:
            logger.error(f"時間別DiCE提案生成で予期しないエラー: {e}", exc_info=True)
            return self._get_error_hourly_schedule(str(e))
    
    def _generate_hourly_alternative(self, original_activity: pd.Series, 
                                   predictor, hour: int) -> dict:
        """
        特定の時間の活動に対する代替提案を生成
        """
        try:
            # 現在の活動でのフラストレーション予測
            current_frustration = predictor.predict_single_activity(
                original_activity.get('CatSub', 'その他'),
                original_activity.get('Duration', 60),
                original_activity['Timestamp']
            )['predicted_frustration']
            
            # 時間帯に適した代替活動候補
            alternative_activities = self._get_time_appropriate_activities(hour)
            
            best_alternative = None
            best_improvement = 0
            
            # 各代替活動での予測を実行
            for alt_activity in alternative_activities:
                alt_prediction = predictor.predict_single_activity(
                    alt_activity,
                    original_activity.get('Duration', 60),
                    original_activity['Timestamp']
                )
                
                improvement = current_frustration - alt_prediction['predicted_frustration']
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_alternative = alt_activity
            
            if best_alternative and best_improvement > 1:
                return {
                    'hour': hour,
                    'time_range': f"{hour:02d}:00-{hour+1:02d}:00",
                    'original_activity': original_activity.get('CatSub', '不明'),
                    'original_frustration': current_frustration,
                    'suggested_activity': best_alternative,
                    'suggested_frustration': current_frustration - best_improvement,
                    'improvement': best_improvement,
                    'reason': self._get_improvement_reason(original_activity.get('CatSub'), best_alternative),
                    'confidence': min(0.9, 0.7 + best_improvement * 0.1)
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"時間別代替活動生成エラー: {e}")
            return None
    
    def _get_time_appropriate_activities(self, hour: int) -> List[str]:
        """
        時間帯に適した活動候補を取得
        """
        if 6 <= hour < 9:
            return ['朝食', '軽い運動', '散歩', 'ストレッチ', '読書']
        elif 9 <= hour < 12:
            return ['作業', '家事', '買い物', '散歩', 'リラックス']
        elif 12 <= hour < 14:
            return ['昼食', '昼休み', '散歩', 'リラックス']
        elif 14 <= hour < 18:
            return ['作業', '運動', '散歩', '家事', '趣味']
        elif 18 <= hour < 21:
            return ['夕食', '入浴', 'リラックス', '軽い運動', '趣味']
        elif 21 <= hour < 24:
            return ['入浴', 'リラックス', '読書', 'ストレッチ', '睡眠準備']
        else:
            return ['睡眠', 'リラックス']
    
    def _recommend_activity_for_hour(self, hour: int) -> str:
        """
        時間帯に最も適した推奨活動を取得
        """
        recommendations = {
            6: '朝の散歩', 7: '朝食', 8: 'ストレッチ',
            9: '作業開始', 10: '集中作業', 11: '軽い休憩',
            12: '昼食', 13: '昼休み', 14: '午後の作業',
            15: '軽い運動', 16: '作業継続', 17: '夕方の散歩',
            18: '夕食準備', 19: '夕食', 20: 'リラックス',
            21: '入浴', 22: '読書', 23: '睡眠準備'
        }
        
        if hour in recommendations:
            return recommendations[hour]
        elif 0 <= hour < 6:
            return '睡眠'
        else:
            return 'リラックス'
    
    def _get_improvement_reason(self, original: str, alternative: str) -> str:
        """
        改善理由の説明を生成
        """
        improvement_reasons = {
            ('仕事', 'リラックス'): 'ストレス軽減のため',
            ('作業', '散歩'): '気分転換と運動のため',
            ('勉強', '軽い運動'): '集中力回復のため',
            ('通学', '読書'): 'より快適な時間の過ごし方',
            ('洗濯', 'リラックス'): '家事ストレス軽減のため',
            ('睡眠', 'ストレッチ'): '睡眠の質向上のため'
        }
        
        key = (original, alternative)
        return improvement_reasons.get(key, f'{alternative}によるストレス軽減効果')
    
    def _get_error_hourly_schedule(self, error_message: str) -> dict:
        """エラー発生時の時間別スケジュール"""
        return {
            'type': 'error',
            'status': 'error',
            'error_message': error_message,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'hourly_schedule': [],
            'total_improvement': 0,
            'message': 'エラーが発生しました',
            'confidence': 0.0,
            'summary': 'DiCEの実行中にエラーが発生しました。システム管理者に連絡してください。',
            'user_message': 'DiCEの実行中にエラーが発生しました。システム管理者に連絡してください。'
        }

    def _get_no_data_hourly_schedule(self, reason: str, details: dict = None) -> dict:
        """データが不足している場合の時間別スケジュール"""
        result = {
            'type': 'no_data',
            'status': 'no_data',
            'reason': reason,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'hourly_schedule': [],
            'total_improvement': 0,
            'message': 'データが不足しています',
            'confidence': 0.1,
            'summary': 'データが不足しているため、詳細な提案を生成できませんでした',
            'user_message': 'データが不足しているため、DiCEによる改善提案を生成できませんでした。'
        }
        if details:
            result['details'] = details
        return result

    def _get_no_solution_hourly_schedule(self, reason: str, details: dict = None) -> dict:
        """改善提案が見つからない場合の時間別スケジュール（データはあるが解なし）"""
        result = {
            'type': 'no_solution',
            'status': 'no_solution',
            'reason': reason,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'hourly_schedule': [],
            'total_improvement': 0,
            'message': '改善提案を生成できませんでした',
            'confidence': 0.3,
            'summary': 'データを分析しましたが、現時点では大きな改善提案を生成できませんでした',
            'user_message': 'データを分析しましたが、現時点では大きな改善提案を生成できませんでした。'
        }
        if details:
            result['details'] = details
        return result

    def _get_fallback_hourly_schedule(self) -> dict:
        """フォールバック用の時間別スケジュール（後方互換性のため保持）"""
        return self._get_no_solution_hourly_schedule("詳細な提案を生成できませんでした")