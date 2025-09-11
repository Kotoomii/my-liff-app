"""
LLMによる自然言語フィードバック生成機能
過去24時間のDiCE結果を考慮して自然言語でフィードバックを生成
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
import requests
import os

from config import Config

logger = logging.getLogger(__name__)

class LLMFeedbackGenerator:
    def __init__(self):
        self.config = Config()
        self.llm_api_key = os.environ.get('OPENAI_API_KEY', '')
        self.llm_api_base = "https://api.openai.com/v1"
        
    def generate_comprehensive_feedback(self, 
                                      dice_results: List[Dict],
                                      user_profile: Dict = None,
                                      feedback_type: str = "evening") -> Dict:
        """
        過去24時間分のDiCE結果を考慮した包括的なフィードバックを生成
        
        Args:
            dice_results: 過去24時間のDiCE結果リスト
            user_profile: ユーザープロフィール（任意）
            feedback_type: "morning" (朝のフィードバック) または "evening" (夕方のフィードバック)
        """
        try:
            if not dice_results:
                return self._get_fallback_feedback(feedback_type)
            
            # DiCE結果を分析してサマリーを作成
            analysis_summary = self._analyze_dice_results(dice_results)
            
            # LLMプロンプトを構築
            prompt = self._build_feedback_prompt(
                analysis_summary, 
                user_profile or {}, 
                feedback_type
            )
            
            # LLMからフィードバックを生成
            if self.llm_api_key:
                llm_feedback = self._generate_with_llm(prompt)
            else:
                llm_feedback = self._generate_rule_based_feedback(analysis_summary, feedback_type)
            
            return {
                'type': 'comprehensive_feedback',
                'feedback_type': feedback_type,
                'generated_at': datetime.now().isoformat(),
                'content': llm_feedback,
                'analysis_summary': analysis_summary,
                'num_dice_results': len(dice_results),
                'confidence': 0.85 if self.llm_api_key else 0.65
            }
            
        except Exception as e:
            logger.error(f"包括的フィードバック生成エラー: {e}")
            return self._get_fallback_feedback(feedback_type)
    
    def _analyze_dice_results(self, dice_results: List[Dict]) -> Dict:
        """
        DiCE結果を分析してサマリーを作成
        """
        try:
            total_improvement = 0
            total_opportunities = 0
            activity_improvements = {}
            time_period_improvements = {'朝': 0, '午後': 0, '夕方': 0, '夜': 0}
            top_problematic_activities = {}
            top_recommended_activities = {}
            
            for result in dice_results:
                if result.get('type') == 'activity_counterfactual':
                    total_improvement += result.get('total_improvement', 0)
                    total_opportunities += result.get('num_suggestions', 0)
                    
                    # タイムライン分析
                    for timeline_item in result.get('timeline', []):
                        hour = timeline_item.get('hour', 12)
                        time_period = self._get_time_period(hour)
                        time_period_improvements[time_period] += timeline_item.get('frustration_reduction', 0)
                        
                        # 活動別改善ポテンシャル
                        orig_activity = timeline_item.get('original_activity', '不明')
                        sugg_activity = timeline_item.get('suggested_activity', '不明')
                        
                        if orig_activity not in activity_improvements:
                            activity_improvements[orig_activity] = 0
                        activity_improvements[orig_activity] += timeline_item.get('frustration_reduction', 0)
                        
                        # 問題のある活動と推奨活動をカウント
                        top_problematic_activities[orig_activity] = top_problematic_activities.get(orig_activity, 0) + 1
                        top_recommended_activities[sugg_activity] = top_recommended_activities.get(sugg_activity, 0) + 1
            
            # 上位3つを取得
            top_problematic = sorted(top_problematic_activities.items(), key=lambda x: x[1], reverse=True)[:3]
            top_recommended = sorted(top_recommended_activities.items(), key=lambda x: x[1], reverse=True)[:3]
            best_time_period = max(time_period_improvements.items(), key=lambda x: x[1])
            
            return {
                'total_improvement_potential': total_improvement,
                'total_opportunities': total_opportunities,
                'average_improvement': total_improvement / max(1, total_opportunities),
                'best_improvement_time_period': best_time_period,
                'time_period_breakdown': time_period_improvements,
                'top_problematic_activities': [item[0] for item in top_problematic],
                'top_recommended_activities': [item[0] for item in top_recommended],
                'activity_improvements': activity_improvements,
                'overall_stress_level': 'high' if total_improvement > 80 else 'medium' if total_improvement > 30 else 'low'
            }
            
        except Exception as e:
            logger.error(f"DiCE結果分析エラー: {e}")
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
    
    def _build_feedback_prompt(self, 
                              analysis_summary: Dict, 
                              user_profile: Dict, 
                              feedback_type: str) -> str:
        """
        LLM用のプロンプトを構築
        """
        try:
            # 基本情報
            total_improvement = analysis_summary.get('total_improvement_potential', 0)
            opportunities = analysis_summary.get('total_opportunities', 0)
            best_time = analysis_summary.get('best_improvement_time_period', ('不明', 0))
            stress_level = analysis_summary.get('overall_stress_level', 'medium')
            
            # 時間帯情報
            if feedback_type == "morning":
                time_context = "朝のフィードバック（1日の始まりに向けて）"
                goal = "今日一日を快適に過ごすための具体的なアドバイス"
            else:
                time_context = "夕方のフィードバック（1日の振り返りと明日への準備）"
                goal = "今日の行動を振り返り、明日に向けた改善提案"
            
            prompt = f"""
あなたはストレス管理の専門家です。以下の分析結果をもとに、温かみのある自然な日本語で具体的で実践的なフィードバックを生成してください。

## コンテキスト
- {time_context}
- 目的: {goal}

## 分析結果
- 総改善ポテンシャル: {total_improvement:.1f}点
- 改善機会の数: {opportunities}個
- 全体的なストレスレベル: {stress_level}
- 最も改善余地がある時間帯: {best_time[0]}（{best_time[1]:.1f}点の改善ポテンシャル）

### 問題となりやすい活動
{', '.join(analysis_summary.get('top_problematic_activities', []))}

### 推奨される活動
{', '.join(analysis_summary.get('top_recommended_activities', []))}

### 時間帯別改善ポテンシャル
- 朝: {analysis_summary.get('time_period_breakdown', {}).get('朝', 0):.1f}点
- 午後: {analysis_summary.get('time_period_breakdown', {}).get('午後', 0):.1f}点
- 夕方: {analysis_summary.get('time_period_breakdown', {}).get('夕方', 0):.1f}点
- 夜: {analysis_summary.get('time_period_breakdown', {}).get('夜', 0):.1f}点

## フィードバック生成の指針
1. **温かく支援的な口調**で書いてください
2. **具体的で実践可能な提案**を含めてください
3. **数値を自然に組み込んで**説得力を持たせてください
4. **時間帯に応じた適切なアドバイス**を提供してください
5. **前向きで励ましのメッセージ**で終わってください

フィードバックは200-300文字程度で、3-4つのパラグラフに分けて生成してください。
"""
            
            return prompt
            
        except Exception as e:
            logger.error(f"プロンプト構築エラー: {e}")
            return "システムエラーが発生しました。基本的なアドバイスを提供します。"
    
    def _generate_with_llm(self, prompt: str) -> str:
        """
        OpenAI API等のLLMを使用してフィードバックを生成
        """
        try:
            headers = {
                'Authorization': f'Bearer {self.llm_api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': 'gpt-3.5-turbo',
                'messages': [
                    {
                        'role': 'system',
                        'content': 'あなたは優秀なストレス管理コンサルタントです。温かく、具体的で実践的なアドバイスを提供します。'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'max_tokens': 500,
                'temperature': 0.7
            }
            
            response = requests.post(
                f"{self.llm_api_base}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                logger.warning(f"LLM API エラー: {response.status_code}")
                return self._generate_rule_based_feedback_simple()
                
        except Exception as e:
            logger.error(f"LLM フィードバック生成エラー: {e}")
            return self._generate_rule_based_feedback_simple()
    
    def _generate_rule_based_feedback(self, analysis_summary: Dict, feedback_type: str) -> str:
        """
        ルールベースでフィードバックを生成（LLMが利用できない場合）
        """
        try:
            total_improvement = analysis_summary.get('total_improvement_potential', 0)
            opportunities = analysis_summary.get('total_opportunities', 0)
            best_time = analysis_summary.get('best_improvement_time_period', ('不明', 0))
            stress_level = analysis_summary.get('overall_stress_level', 'medium')
            
            feedback_parts = []
            
            # 時間帯に応じた挨拶
            if feedback_type == "morning":
                greeting = "おはようございます！"
                context = "今日一日を快適に過ごすために、"
            else:
                greeting = "お疲れ様でした！"
                context = "今日の行動を振り返って、"
            
            feedback_parts.append(greeting)
            
            # 全体的な評価
            if stress_level == 'high' and total_improvement > 50:
                assessment = f"{context}昨日は{opportunities}個の改善機会があり、{total_improvement:.1f}点のストレス軽減が期待できそうでした。"
            elif stress_level == 'medium':
                assessment = f"{context}バランスの取れた一日でしたが、{total_improvement:.1f}点程度の改善余地がありそうです。"
            else:
                assessment = f"{context}ストレス管理が上手にできていますね。"
            
            feedback_parts.append(assessment)
            
            # 具体的なアドバイス
            if best_time[1] > 15:
                advice = f"特に{best_time[0]}の時間帯に気をつけて、"
                
                time_specific_advice = {
                    '朝': "朝の時間にゆとりを持ち、深呼吸や軽いストレッチを取り入れてみてください。",
                    '午後': "午後は小まめに休憩を取り、水分補給を心がけましょう。",
                    '夕方': "夕方は一日の疲れが溜まる時間です。軽い運動や好きな音楽でリフレッシュを。",
                    '夜': "夜はリラックスタイムを大切に、入浴や読書で心を落ち着けましょう。"
                }
                
                advice += time_specific_advice.get(best_time[0], "リラックスできる活動を取り入れてみてください。")
            else:
                advice = "現在のペースを維持しつつ、無理をしすぎないよう注意してくださいね。"
            
            feedback_parts.append(advice)
            
            # 励ましのメッセージ
            if feedback_type == "morning":
                encouragement = "今日も素敵な一日になりますように！"
            else:
                encouragement = "明日はもっと快適な一日になることを願っています。"
            
            feedback_parts.append(encouragement)
            
            return " ".join(feedback_parts)
            
        except Exception as e:
            logger.error(f"ルールベースフィードバック生成エラー: {e}")
            return self._generate_rule_based_feedback_simple()
    
    def _generate_rule_based_feedback_simple(self) -> str:
        """
        シンプルなフォールバックフィードバック
        """
        return ("今日もお疲れ様でした。ストレス管理において、小さな変化の積み重ねが大きな違いを生み出します。"
                "十分な休息を取り、好きな活動でリラックスする時間を大切にしてください。"
                "明日はより快適な一日になることを願っています。")
    
    def generate_morning_briefing(self, 
                                yesterday_dice_results: List[Dict],
                                today_schedule: List[Dict] = None) -> Dict:
        """
        朝のブリーフィング生成（昨日の分析 + 今日の予定を考慮）
        """
        try:
            # 昨日の分析
            yesterday_analysis = self._analyze_dice_results(yesterday_dice_results)
            
            # 朝のフィードバックを生成
            morning_feedback = self.generate_comprehensive_feedback(
                yesterday_dice_results, 
                feedback_type="morning"
            )
            
            # 今日の予定に基づく追加アドバイス
            schedule_advice = []
            if today_schedule:
                schedule_advice = self._generate_schedule_based_advice(
                    today_schedule, yesterday_analysis
                )
            
            return {
                'type': 'morning_briefing',
                'generated_at': datetime.now().isoformat(),
                'yesterday_summary': yesterday_analysis,
                'main_feedback': morning_feedback['content'],
                'schedule_advice': schedule_advice,
                'key_recommendations': self._extract_key_recommendations(yesterday_analysis),
                'confidence': morning_feedback.get('confidence', 0.7)
            }
            
        except Exception as e:
            logger.error(f"朝のブリーフィング生成エラー: {e}")
            return self._get_fallback_morning_briefing()
    
    def generate_evening_summary(self, today_dice_results: List[Dict]) -> Dict:
        """
        夕方のサマリー生成（今日の振り返り）
        """
        try:
            # 今日の分析
            today_analysis = self._analyze_dice_results(today_dice_results)
            
            # 夕方のフィードバックを生成
            evening_feedback = self.generate_comprehensive_feedback(
                today_dice_results,
                feedback_type="evening"
            )
            
            # 今日の成果と改善点
            achievements = self._identify_achievements(today_analysis)
            improvement_areas = self._identify_improvement_areas(today_analysis)
            
            return {
                'type': 'evening_summary',
                'generated_at': datetime.now().isoformat(),
                'today_summary': today_analysis,
                'main_feedback': evening_feedback['content'],
                'achievements': achievements,
                'improvement_areas': improvement_areas,
                'tomorrow_recommendations': self._generate_tomorrow_recommendations(today_analysis),
                'confidence': evening_feedback.get('confidence', 0.7)
            }
            
        except Exception as e:
            logger.error(f"夕方のサマリー生成エラー: {e}")
            return self._get_fallback_evening_summary()
    
    def _generate_schedule_based_advice(self, 
                                      today_schedule: List[Dict], 
                                      yesterday_analysis: Dict) -> List[str]:
        """
        今日のスケジュールに基づく追加アドバイス
        """
        advice = []
        
        try:
            problematic_activities = yesterday_analysis.get('top_problematic_activities', [])
            
            for schedule_item in today_schedule:
                activity = schedule_item.get('activity', '')
                start_time = schedule_item.get('start_time', '')
                
                if activity in problematic_activities:
                    advice.append(f"{start_time} {activity}: 昨日ストレスが高めでした。事前の準備と休憩を心がけてください。")
                elif '会議' in activity or '仕事' in activity:
                    advice.append(f"{start_time} {activity}: 深呼吸をして、集中力を高めましょう。")
                    
            if not advice:
                advice.append("今日のスケジュールを考慮すると、定期的な休憩を取ることをお勧めします。")
                
        except Exception as e:
            logger.error(f"スケジュールベースアドバイス生成エラー: {e}")
            advice = ["今日も一日、無理をせずペースを保って過ごしてくださいね。"]
        
        return advice
    
    def _extract_key_recommendations(self, analysis: Dict) -> List[str]:
        """
        主要な推奨事項を抽出
        """
        recommendations = []
        
        try:
            best_time = analysis.get('best_improvement_time_period', ('', 0))
            if best_time[1] > 10:
                recommendations.append(f"{best_time[0]}の時間帯に特に注意")
            
            recommended_activities = analysis.get('top_recommended_activities', [])
            if recommended_activities:
                recommendations.append(f"推奨活動: {', '.join(recommended_activities[:2])}")
            
            if analysis.get('overall_stress_level') == 'high':
                recommendations.append("今日は特にストレス管理を意識して")
                
        except Exception as e:
            logger.error(f"主要推奨事項抽出エラー: {e}")
        
        return recommendations or ["今日も健康的な一日を過ごしましょう"]
    
    def _identify_achievements(self, analysis: Dict) -> List[str]:
        """
        今日の成果を特定
        """
        achievements = []
        
        if analysis.get('overall_stress_level') == 'low':
            achievements.append("今日は全体的にストレス管理が上手にできていました")
        
        improvement_potential = analysis.get('total_improvement_potential', 0)
        if improvement_potential < 30:
            achievements.append("バランスの取れた一日を過ごせました")
        
        return achievements or ["今日も一日お疲れさまでした"]
    
    def _identify_improvement_areas(self, analysis: Dict) -> List[str]:
        """
        改善領域を特定
        """
        areas = []
        
        best_time = analysis.get('best_improvement_time_period', ('', 0))
        if best_time[1] > 15:
            areas.append(f"{best_time[0]}の時間帯での活動見直し")
        
        problematic_activities = analysis.get('top_problematic_activities', [])
        if problematic_activities:
            areas.append(f"{', '.join(problematic_activities[:2])}での休憩増加")
        
        return areas or ["現在のペースを維持しましょう"]
    
    def _generate_tomorrow_recommendations(self, analysis: Dict) -> List[str]:
        """
        明日への推奨事項を生成
        """
        recommendations = []
        
        recommended_activities = analysis.get('top_recommended_activities', [])
        if recommended_activities:
            recommendations.append(f"明日は{', '.join(recommended_activities[:2])}を取り入れてみてください")
        
        best_time = analysis.get('best_improvement_time_period', ('', 0))
        if best_time[1] > 10:
            recommendations.append(f"{best_time[0]}の時間帯は特に意識的にリラックスを")
        
        return recommendations or ["明日も健康的な一日を過ごしてください"]
    
    def _get_fallback_feedback(self, feedback_type: str) -> Dict:
        """フォールバック用フィードバック"""
        content = self._generate_rule_based_feedback_simple()
        
        return {
            'type': 'comprehensive_feedback',
            'feedback_type': feedback_type,
            'generated_at': datetime.now().isoformat(),
            'content': content,
            'analysis_summary': {},
            'confidence': 0.3
        }
    
    def _get_fallback_morning_briefing(self) -> Dict:
        """フォールバック用朝のブリーフィング"""
        return {
            'type': 'morning_briefing',
            'generated_at': datetime.now().isoformat(),
            'main_feedback': "おはようございます！今日も健康的で快適な一日を過ごしましょう。",
            'key_recommendations': ["規則正しい生活リズムを心がけましょう"],
            'confidence': 0.3
        }
    
    def _get_fallback_evening_summary(self) -> Dict:
        """フォールバック用夕方のサマリー"""
        return {
            'type': 'evening_summary',
            'generated_at': datetime.now().isoformat(),
            'main_feedback': "今日もお疲れさまでした。ゆっくり休んで、明日に備えてください。",
            'achievements': ["今日も一日お疲れさまでした"],
            'tomorrow_recommendations': ["明日も健康的な一日を過ごしてください"],
            'confidence': 0.3
        }