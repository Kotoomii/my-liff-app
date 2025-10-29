"""
LLMによる自然言語フィードバック生成機能
過去24時間のDiCE結果を考慮して自然言語でフィードバックを生成
Google Cloud Secret Managerを使用してAPIキーを安全に管理
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
        logger.info("=" * 60)
        logger.info("🚀 LLMFeedbackGenerator 初期化開始")
        logger.info("=" * 60)

        self.config = Config()
        logger.info(f"📋 設定読み込み完了 (IS_CLOUD_RUN: {self.config.IS_CLOUD_RUN})")

        self.llm_api_key = self._get_api_key_from_secret_manager()
        self.llm_api_base = "https://api.openai.com/v1"

        if self.llm_api_key:
            logger.info(f"✅ LLMFeedbackGenerator 初期化完了 (APIキー: 設定済み)")
        else:
            logger.warning(f"⚠️ LLMFeedbackGenerator 初期化完了 (APIキー: 未設定)")
        logger.info("=" * 60)

    def _get_api_key_from_secret_manager(self) -> str:
        """
        環境変数からOpenAI APIキーを取得
        Cloud Run環境ではSecret Managerのシークレットが環境変数としてマウントされます
        """
        logger.info("🔑 OpenAI APIキー取得を開始...")
        logger.info("📍 環境変数 'OPENAI_API_KEY' を確認中...")

        try:
            # OPENAI_API_KEY または OPEN_API_KEY から取得（両方に対応）
            api_key = os.environ.get('OPENAI_API_KEY', os.environ.get('OPEN_API_KEY', ''))

            if api_key:
                # セキュリティのため最初の7文字のみ表示
                masked_key = api_key[:7] + "..." if len(api_key) > 7 else "***"
                logger.info(f"✅ OpenAI APIキーを環境変数から取得しました")
                logger.info(f"🔐 APIキー (マスク表示): {masked_key}")
                logger.info(f"📏 APIキーの長さ: {len(api_key)}文字")
            else:
                logger.error("❌ OPENAI_API_KEY環境変数が設定されていません！")
                if self.config.IS_CLOUD_RUN:
                    logger.error("💡 Cloud Run環境: Secret Managerのシークレットを環境変数としてマウントしてください")
                    logger.error("   例: gcloud run services update SERVICE_NAME --update-secrets=OPENAI_API_KEY=openai-api-key:latest")
                else:
                    logger.error("💡 ローカル環境: ターミナルで 'export OPENAI_API_KEY=your-api-key' を実行してください")

            return api_key

        except Exception as e:
            logger.error(f"APIキー取得エラー: {e}")
            return ''
        
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
                logger.info("🔑 OpenAI APIキーが設定されています。ChatGPTでフィードバックを生成します。")
                llm_feedback = self._generate_with_llm(prompt)
            else:
                logger.warning("⚠️ OpenAI APIキーが設定されていません。フォールバックメッセージを使用します。")
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
2. **具体的で実践可能な提案**を1-2つ含めてください
3. **数値を自然に組み込んで**説得力を持たせてください
4. **励ましと前向きなメッセージ**で締めくくってください
5. **最後に必ず「※ChatGPTによって生成されたアドバイスです。」という文言を追加**してください

フィードバックは100文字程度で、簡潔で親しみやすい言葉遣いで書いてください。
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
            logger.info("🤖 ChatGPT API (gpt-3.5-turbo) を呼び出し中...")
            logger.debug(f"📤 送信するプロンプト: {prompt[:200]}...")  # 最初の200文字のみ

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
                generated_text = result['choices'][0]['message']['content'].strip()
                logger.info(f"✅ ChatGPT APIからフィードバックを生成しました (文字数: {len(generated_text)})")
                logger.info(f"📝 生成されたフィードバック: {generated_text}")
                return generated_text
            else:
                logger.warning(f"❌ LLM API エラー: {response.status_code}, Response: {response.text}")
                return self._generate_rule_based_feedback_simple()
                
        except Exception as e:
            logger.error(f"LLM フィードバック生成エラー: {e}")
            return self._generate_rule_based_feedback_simple()
    
    def _generate_rule_based_feedback(self, analysis_summary: Dict, feedback_type: str) -> str:
        """
        APIキーがない場合のシンプルなフォールバックメッセージ
        """
        return "フィードバックを生成するにはOpenAI APIキーが必要です。"
    
    def _generate_rule_based_feedback_simple(self) -> str:
        """
        シンプルなフォールバックフィードバック
        """
        return "フィードバックを生成するにはOpenAI APIキーが必要です。"
    
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

    def generate_daily_dice_feedback(self,
                                    daily_dice_result: Dict,
                                    timeline_data: List[Dict] = None) -> Dict:
        """
        1日の終わりにDiCE結果に基づいた日次フィードバックを生成
        タイムライン全体を考慮した包括的なアドバイスを提供

        Args:
            daily_dice_result: 1日分のDiCE分析結果
            timeline_data: 1日のタイムラインデータ（オプション）

        Returns:
            日次フィードバック辞書
        """
        try:
            if not daily_dice_result:
                return self._get_fallback_daily_feedback()

            # DiCE結果から重要な情報を抽出
            hourly_schedule = daily_dice_result.get('hourly_schedule', [])
            total_improvement = daily_dice_result.get('total_improvement', 0)
            date = daily_dice_result.get('date', datetime.now().strftime('%Y-%m-%d'))

            # タイムラインデータから統計情報を計算
            timeline_stats = self._analyze_timeline_data(timeline_data) if timeline_data else {}

            # プロンプトを構築
            prompt = self._build_daily_dice_feedback_prompt(
                hourly_schedule,
                total_improvement,
                date,
                timeline_stats
            )

            # LLMでフィードバック生成
            if self.llm_api_key:
                logger.info("🔑 OpenAI APIキーが設定されています。ChatGPTでフィードバックを生成します。")
                feedback_content = self._generate_with_llm(prompt)
            else:
                logger.warning("⚠️ OpenAI APIキーが設定されていません。フォールバックメッセージを使用します。")
                feedback_content = self._generate_rule_based_daily_feedback(
                    hourly_schedule,
                    total_improvement,
                    timeline_stats
                )

            # 明日へのアクションプランを生成
            action_plan = self._generate_tomorrow_action_plan(hourly_schedule, timeline_stats)

            return {
                'type': 'daily_dice_feedback',
                'date': date,
                'generated_at': datetime.now().isoformat(),
                'main_feedback': feedback_content,
                'total_improvement_potential': total_improvement,
                'num_suggestions': len(hourly_schedule),
                'action_plan': action_plan,
                'timeline_stats': timeline_stats,
                'confidence': 0.85 if self.llm_api_key else 0.65
            }

        except Exception as e:
            logger.error(f"日次DiCEフィードバック生成エラー: {e}")
            return self._get_fallback_daily_feedback()

    def _analyze_timeline_data(self, timeline_data: List[Dict]) -> Dict:
        """
        タイムラインデータを分析して統計情報を生成
        """
        try:
            if not timeline_data:
                return {}

            frustration_values = [item.get('frustration_value', 10) for item in timeline_data]
            activities = [item.get('activity', '不明') for item in timeline_data]

            # 活動別の平均フラストレーション値
            activity_frustration = {}
            for item in timeline_data:
                activity = item.get('activity', '不明')
                frustration = item.get('frustration_value', 10)
                if activity not in activity_frustration:
                    activity_frustration[activity] = []
                activity_frustration[activity].append(frustration)

            # 平均値を計算
            activity_avg = {
                activity: sum(values) / len(values)
                for activity, values in activity_frustration.items()
            }

            # 最もストレスが高かった活動と低かった活動
            sorted_activities = sorted(activity_avg.items(), key=lambda x: x[1], reverse=True)

            return {
                'avg_frustration': sum(frustration_values) / len(frustration_values) if frustration_values else 10,
                'min_frustration': min(frustration_values) if frustration_values else 0,
                'max_frustration': max(frustration_values) if frustration_values else 20,
                'total_activities': len(timeline_data),
                'highest_stress_activity': sorted_activities[0] if sorted_activities else ('不明', 10),
                'lowest_stress_activity': sorted_activities[-1] if sorted_activities else ('不明', 10),
                'activity_distribution': activity_avg
            }

        except Exception as e:
            logger.error(f"タイムラインデータ分析エラー: {e}")
            return {}

    def _build_daily_dice_feedback_prompt(self,
                                         hourly_schedule: List[Dict],
                                         total_improvement: float,
                                         date: str,
                                         timeline_stats: Dict) -> str:
        """
        日次DiCEフィードバック用のプロンプトを構築
        """
        try:
            # 改善提案をテキスト化
            suggestions_text = []
            for suggestion in hourly_schedule[:5]:  # 上位5件
                time_range = suggestion.get('time_range', '不明')
                original = suggestion.get('original_activity', '不明')
                suggested = suggestion.get('suggested_activity', '不明')
                improvement = suggestion.get('improvement', 0)

                suggestions_text.append(
                    f"- {time_range}: 「{original}」→「{suggested}」(改善: {improvement:.1f}点)"
                )

            # タイムライン統計
            avg_frustration = timeline_stats.get('avg_frustration', 10)
            highest_stress = timeline_stats.get('highest_stress_activity', ('不明', 10))
            lowest_stress = timeline_stats.get('lowest_stress_activity', ('不明', 10))

            prompt = f"""
あなたはストレス管理の専門家です。今日1日の活動データとDiCE分析結果をもとに、ユーザーに寄り添った温かみのあるフィードバックを生成してください。

## 今日の日付
{date}

## 今日の統計
- 平均フラストレーション値: {avg_frustration:.1f}点 (1-20スケール)
- 最大: {timeline_stats.get('max_frustration', 20):.1f}点、最小: {timeline_stats.get('min_frustration', 0):.1f}点
- 活動数: {timeline_stats.get('total_activities', 0)}件
- 最もストレスが高かった活動: {highest_stress[0]} ({highest_stress[1]:.1f}点)
- 最もリラックスできた活動: {lowest_stress[0]} ({lowest_stress[1]:.1f}点)

## DiCE分析による改善提案
総改善ポテンシャル: {total_improvement:.1f}点
提案数: {len(hourly_schedule)}件

### 主な改善提案
{chr(10).join(suggestions_text[:5]) if suggestions_text else '改善提案なし'}

## フィードバック生成の指針
1. **今日1日の振り返り**から始めてください（ポジティブな点を強調）
2. **数値を自然に織り交ぜて**具体性を持たせてください
3. **DiCE提案から1-2つの実践的なアドバイス**を含めてください
4. **励ましと前向きなメッセージ**で締めくくってください
5. **最後に必ず「※ChatGPTによって生成されたアドバイスです。」という文言を追加**してください
6. **【テスト用】文章の最後に山手線の駅名を一つランダムに追加してください**（例：「渋谷」「新宿」「東京」など）

フィードバックは100文字程度で、簡潔で親しみやすい言葉遣いで書いてください。
"""

            return prompt

        except Exception as e:
            logger.error(f"日次DiCEプロンプト構築エラー: {e}")
            return "今日もお疲れさまでした。明日はより良い一日になりますように。"

    def _generate_rule_based_daily_feedback(self,
                                           hourly_schedule: List[Dict],
                                           total_improvement: float,
                                           timeline_stats: Dict) -> str:
        """
        APIキーがない場合のシンプルなフォールバックメッセージ
        """
        return "今日もお疲れさまでした。フィードバックを生成するにはOpenAI APIキーが必要です。"

    def _generate_tomorrow_action_plan(self,
                                      hourly_schedule: List[Dict],
                                      timeline_stats: Dict) -> List[str]:
        """
        明日のアクションプランを生成
        """
        try:
            action_plan = []

            # 改善効果が高い上位3件の提案を抽出
            top_suggestions = sorted(
                hourly_schedule,
                key=lambda x: x.get('improvement', 0),
                reverse=True
            )[:3]

            for suggestion in top_suggestions:
                time_range = suggestion.get('time_range', '不明')
                suggested = suggestion.get('suggested_activity', '不明')
                improvement = suggestion.get('improvement', 0)

                if improvement > 2:  # 2点以上の改善効果がある場合のみ
                    action_plan.append(
                        f"{time_range}頃に「{suggested}」を試してみる (期待効果: {improvement:.1f}点)"
                    )

            # 一般的なアドバイスを追加
            highest_stress = timeline_stats.get('highest_stress_activity', ('不明', 10))
            if highest_stress[1] > 15:
                action_plan.append(f"「{highest_stress[0]}」の前後に休憩時間を設ける")

            if not action_plan:
                action_plan.append("現在の良好な生活リズムを維持する")
                action_plan.append("定期的な休憩とリラックスタイムを確保する")

            return action_plan

        except Exception as e:
            logger.error(f"明日のアクションプラン生成エラー: {e}")
            return ["十分な睡眠と休息を取る", "無理のないペースで活動する"]

    def _get_fallback_daily_feedback(self) -> Dict:
        """フォールバック用日次フィードバック"""
        return {
            'type': 'daily_dice_feedback',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'generated_at': datetime.now().isoformat(),
            'main_feedback': "今日もお疲れさまでした。ゆっくり休んで、明日も健康的な一日を過ごしてください。",
            'total_improvement_potential': 0,
            'num_suggestions': 0,
            'action_plan': ["十分な休息を取る", "明日も無理をしない"],
            'timeline_stats': {},
            'confidence': 0.3
        }