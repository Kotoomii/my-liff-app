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
    def __init__(self, sheets_connector=None):
        logger.info("=" * 60)
        logger.info("🚀 LLMFeedbackGenerator 初期化開始")
        logger.info("=" * 60)

        self.config = Config()
        logger.info(f"📋 設定読み込み完了 (IS_CLOUD_RUN: {self.config.IS_CLOUD_RUN})")

        self.sheets_connector = sheets_connector

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
                        'content': 'あなたは優秀なストレス管理コンサルタントです。温かく、具体的で実践的なアドバイスを提供します。ユーザーの自律性を尊重し、命令形（「〜しましょう」「〜してください」）は絶対に使わず、提案型の表現（「〜してみるのはいかがでしょうか？」など）のみを使用してください。決定権は常にユーザーにあります。'
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

    def _generate_rule_based_feedback_simple(self) -> str:
        """
        シンプルなフォールバックフィードバック
        """
        return "フィードバックを生成するにはOpenAI APIキーが必要です。"

    def generate_daily_dice_feedback(self,
                                    daily_dice_result: Dict,
                                    timeline_data: List[Dict] = None,
                                    user_id: str = 'default') -> Dict:
        """
        1日の終わりにDiCE結果に基づいた日次フィードバックを生成
        タイムライン全体を考慮した包括的なアドバイスを提供

        Args:
            daily_dice_result: 1日分のDiCE分析結果
            timeline_data: 1日のタイムラインデータ（オプション）
            user_id: ユーザーID（昨日のデータ取得用、デフォルト: 'default'）

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

            # 昨日のDaily Summaryデータを取得（進捗追跡のため）
            yesterday_summary = None
            if self.sheets_connector:
                from datetime import datetime as dt_class, timedelta
                yesterday_date = (dt_class.strptime(date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
                yesterday_summary = self.sheets_connector.get_daily_summary(user_id, yesterday_date)
                logger.info(f"📊 昨日のデータ取得: {yesterday_date}, 存在={yesterday_summary is not None}")

            # プロンプトを構築（昨日のデータを含む）
            prompt = self._build_daily_dice_feedback_prompt(
                hourly_schedule,
                total_improvement,
                date,
                timeline_stats,
                yesterday_summary
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

            # frustration_valueがnullでないものだけをフィルタ
            frustration_values = [
                item.get('frustration_value')
                for item in timeline_data
                if item.get('frustration_value') is not None
            ]
            activities = [item.get('activity', '不明') for item in timeline_data]

            # frustration_valueがnullの場合はスキップ
            if len(frustration_values) == 0:
                return {
                    'avg_frustration': None,
                    'min_frustration': None,
                    'max_frustration': None,
                    'total_activities': 0,
                    'highest_stress_activity': ('不明', None),
                    'lowest_stress_activity': ('不明', None),
                    'activity_distribution': {}
                }

            # 活動別の平均フラストレーション値
            activity_frustration = {}
            for item in timeline_data:
                activity = item.get('activity', '不明')
                frustration = item.get('frustration_value')
                # nullの場合はスキップ
                if frustration is None:
                    continue
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
                'avg_frustration': sum(frustration_values) / len(frustration_values) if frustration_values else None,
                'min_frustration': min(frustration_values) if frustration_values else None,
                'max_frustration': max(frustration_values) if frustration_values else None,
                'total_activities': len(frustration_values),  # 予測値があるものだけカウント
                'highest_stress_activity': sorted_activities[0] if sorted_activities else ('不明', None),
                'lowest_stress_activity': sorted_activities[-1] if sorted_activities else ('不明', None),
                'activity_distribution': activity_avg
            }

        except Exception as e:
            logger.error(f"タイムラインデータ分析エラー: {e}")
            return {}

    def _build_daily_dice_feedback_prompt(self,
                                         hourly_schedule: List[Dict],
                                         total_improvement: float,
                                         date: str,
                                         timeline_stats: Dict,
                                         yesterday_summary: Dict = None) -> str:
        """
        日次DiCEフィードバック用のプロンプトを構築
        """
        try:
            # 改善提案をテキスト化
            suggestions_text = []
            for suggestion in hourly_schedule[:5]:  # 上位5件
                time_range = suggestion.get('time_range', suggestion.get('time', '不明'))
                original = suggestion.get('original_activity', '不明')
                suggested = suggestion.get('suggested_activity', '不明')
                improvement = suggestion.get('improvement', 0)

                suggestions_text.append(
                    f"- {time_range}: 「{original}」→「{suggested}」(改善: {improvement:.1f}点)"
                )

            # タイムライン統計
            avg_frustration = timeline_stats.get('avg_frustration')
            highest_stress = timeline_stats.get('highest_stress_activity', ('不明', None))
            lowest_stress = timeline_stats.get('lowest_stress_activity', ('不明', None))

            # データが全くない場合の早期リターン
            if avg_frustration is None:
                return "今日のフラストレーション予測データがありません。Fitbitデータが不足している可能性があります。"

            # 昨日との比較情報を構築
            comparison_text = ""
            if yesterday_summary:
                yesterday_avg = yesterday_summary.get('avg_predicted')
                yesterday_activities = yesterday_summary.get('total_activities', 0)

                if yesterday_avg is not None and avg_frustration is not None:
                    diff = avg_frustration - yesterday_avg
                    diff_direction = "改善" if diff < 0 else "上昇" if diff > 0 else "横ばい"
                    comparison_text = f"""
## 昨日との比較（進捗追跡）
- 昨日の平均フラストレーション値: {yesterday_avg:.1f}点
- 今日の平均: {avg_frustration:.1f}点（昨日より{abs(diff):.1f}点{diff_direction}）
- 活動数: 昨日{yesterday_activities}件 → 今日{timeline_stats.get('total_activities', 0)}件
"""
            else:
                comparison_text = "\n## 昨日との比較\n昨日のデータがありません。初日の記録です。\n"

            # Noneチェックを追加
            max_f = timeline_stats.get('max_frustration')
            min_f = timeline_stats.get('min_frustration')
            highest_stress_val = highest_stress[1] if highest_stress[1] is not None else 0
            lowest_stress_val = lowest_stress[1] if lowest_stress[1] is not None else 0

            prompt = f"""
あなたはストレス管理の専門家です。自己決定理論（Self-Determination Theory）に基づき、ユーザーの自律性を尊重し、内発的動機づけを促すフィードバックを生成してください。

## 今日の日付
{date}
{comparison_text}
## 今日の統計
- 平均フラストレーション値: {avg_frustration:.1f}点 (1-20スケール)
- 最大: {max_f:.1f if max_f is not None else '不明'}点、最小: {min_f:.1f if min_f is not None else '不明'}点
- 活動数: {timeline_stats.get('total_activities', 0)}件
- 最もストレスが高かった活動: {highest_stress[0]} ({highest_stress_val:.1f}点)
- 最もリラックスできた活動: {lowest_stress[0]} ({lowest_stress_val:.1f}点)

## DiCE分析による改善提案
総改善ポテンシャル: {total_improvement:.1f}点
提案数: {len(hourly_schedule)}件

### 主な改善提案
{chr(10).join(suggestions_text[:5]) if suggestions_text else '改善提案なし'}

## フィードバックの必須構造（200文字以内）

以下の内容を必ず含めてください：

### 1. 今日一番ストレスを感じた活動の明示（必須）
- 「今日一番ストレスを感じた活動は「{highest_stress[0]}」でした（{highest_stress_val:.1f}点）。」

### 2. DiCE提案に基づく具体的なアドバイス（必須）
- 上記の主な改善提案から1つ選び、具体的に提案してください
- 例: 「DiCE分析によると、{time_range}に「{suggested}」を取り入れてみるのはいかがでしょうか？（期待効果: {improvement:.1f}点の改善）」

### 3. 自律性の尊重（最重要・必須）
**絶対に使ってはいけない表現（厳守）**:
- ❌ 「〜しましょう」「〜してください」「〜しましょう」
- ❌ 「〜することが大切です」「〜が必要です」「〜すべきです」
- ❌ 「〜を心がけましょう」「〜ように気をつけましょう」

**必ず使うべき提案型の表現**:
- ✅ 「〜してみるのはいかがでしょうか？」
- ✅ 「〜を試してみるのも良いかもしれません」
- ✅ 「もし良ければ〜という選択肢もあります」
- ✅ 「〜も一つの方法かもしれませんね」

**具体例**:
- 良い例: 「{highest_stress[0]}の前後に休憩を挟んでみるのはいかがでしょうか？」
- 悪い例: 「{highest_stress[0]}の後は休憩を取りましょう」

決定権は常にユーザーにあることを忘れないでください。

## DiCE提案の伝え方（重要）
DiCE提案は「もしその時間に別の活動をしていたら、フラストレーションはどう変わるか」という仮定の情報提供です。
実際に活動全体を変えることを推奨するわけではなく、活動選択の影響を知るための参考情報として伝えてください。

**良い伝え方の例**:
- 「もし{time_range}に{original}ではなく{suggested}をしていたら、ストレス値は{improvement:.1f}点低かった可能性があります。」
- 「DiCE分析では、{time_range}の{original}を{suggested}に変えた場合、{improvement:.1f}点の改善が見込まれます。この情報を参考に、明日の活動を考えてみるのも良いかもしれませんね。」

**悪い伝え方の例**:
- 「{time_range}に{suggested}をしてください」（命令形）
- 「{suggested}に変えましょう」（実際に変えることを推奨している）

## 出力形式
1. 今日一番ストレスを感じた活動を明示（活動名と数値）
2. DiCE提案を「もし〜だったら」形式で1つ紹介
3. この情報を参考にすることを提案型で伝える

**文字数**: 200文字以内
**表現**: 禁止表現を絶対に使わず、提案型の表現のみ
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