// スマートミラー専用JavaScript - タッチ操作なし版

class SmartMirrorStressDashboard {
    constructor() {
        this.nasaChart = null;
        this.updateInterval = null;
        this.currentUserId = 'default';
        this.currentDate = new Date().toISOString().split('T')[0];
        this.lastUpdateTime = null;
        this.demoMode = true; // デモモードを有効にする
        this.currentSuggestionIndex = 0; // 現在表示中の提案 (0:A, 1:B)
        this.suggestions = [null, null]; // 提案データ格納
        this.recognition = null; // 音声認識
        this.isListening = false; // 音声認識状態
        console.log('Debug: SmartMirrorStressDashboard initialized with demoMode:', this.demoMode);
        this.init();
    }

    init() {
        this.updateClock();
        this.setAnalysisDate();
        this.setupUserSelector();
        this.setupSuggestionControls();
        this.setupVoiceRecognition();
        this.startDataLoop();
        
        // 時計の更新（毎秒）
        setInterval(() => this.updateClock(), 1000);
        
        // データ自動更新（5分ごと）
        this.updateInterval = setInterval(() => {
            this.loadAllData();
        }, 5 * 60 * 1000);
        
        // 初回データ読み込み
        this.loadAllData();
    }

    updateClock() {
        const now = new Date();
        const timeElement = document.getElementById('current-time');
        const dateElement = document.getElementById('current-date');
        
        if (timeElement) {
            timeElement.textContent = now.toLocaleTimeString('ja-JP', {
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit'
            });
        }
        
        if (dateElement) {
            dateElement.textContent = now.toLocaleDateString('ja-JP', {
                year: 'numeric',
                month: 'long',
                day: 'numeric',
                weekday: 'long'
            });
        }
    }

    setAnalysisDate() {
        const dateDisplay = document.getElementById('analysis-date-display');
        if (dateDisplay) {
            const date = new Date(this.currentDate);
            dateDisplay.textContent = date.toLocaleDateString('ja-JP', {
                month: 'short',
                day: 'numeric'
            });
        }
    }

    startDataLoop() {
        // ユーザーIDを周期的に切り替え（デモ用）
        const users = ['default', 'user1', 'user2'];
        let userIndex = 0;
        
        setInterval(() => {
            userIndex = (userIndex + 1) % users.length;
            this.currentUserId = users[userIndex];
            this.updateUserDisplay();
            this.loadAllData();
        }, 2 * 60 * 1000); // 2分ごとにユーザー切り替え
    }

    updateUserDisplay() {
        const userNameElement = document.getElementById('current-user-name');
        if (userNameElement) {
            const userNames = {
                'default': 'デフォルト',
                'user1': 'ユーザー1',
                'user2': 'ユーザー2'
            };
            userNameElement.textContent = userNames[this.currentUserId] || 'デフォルト';
        }
    }

    async loadAllData() {
        this.updateConnectionStatus('connecting');
        
        try {
            // 並列でデータを読み込み
            await Promise.all([
                this.loadWorkloadData(),
                this.loadTodayOptimalPattern(),
                this.loadTomorrowSuggestions(),
                this.loadDailyAdvice()
            ]);
            
            this.updateConnectionStatus('connected');
            this.updateLastUpdateTime();
            
        } catch (error) {
            console.error('データ読み込みエラー:', error);
            this.updateConnectionStatus('error');
        }
    }

    async loadWorkloadData() {
        try {
            const apiPath = this.demoMode ? '/api/demo/workload/calculate' : '/api/workload/calculate';
            console.log(`Debug: Calling Workload API: ${apiPath} (demoMode: ${this.demoMode})`);
            const response = await fetch(apiPath, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    user_id: this.currentUserId,
                    date: this.currentDate
                })
            });

            const data = await response.json();
            
            if (data.status === 'success') {
                this.updateWorkloadDisplay(data.workload_data);
                this.updateNASAChart(data.workload_data);
                this.updateWellnessMetrics(data.workload_data);
                
                // データ保存インジケーター表示
                if (data.saved_to_sheets) {
                    this.showSaveIndicator();
                }
            } else {
                throw new Error(data.message || '負荷データ取得エラー');
            }
        } catch (error) {
            console.error('負荷データ取得エラー:', error);
            this.showWorkloadError();
        }
    }

    async loadTodayOptimalPattern() {
        try {
            const apiPath = this.demoMode ? '/api/demo/timeline/today' : '/api/timeline/today';
            console.log(`Debug: Calling API: ${apiPath} (demoMode: ${this.demoMode})`);
            const response = await fetch(apiPath, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    user_id: this.currentUserId,
                    date: this.currentDate
                })
            });

            const data = await response.json();
            
            if (data.status === 'success') {
                this.renderOptimalPattern(data.timeline);
                this.updatePatternSummary(data.summary);
            } else {
                throw new Error(data.message || '最適パターン取得エラー');
            }
        } catch (error) {
            console.error('最適パターン取得エラー:', error);
            this.showPatternError();
        }
    }

    async loadTomorrowSuggestions() {
        try {
            const tomorrow = new Date();
            tomorrow.setDate(tomorrow.getDate() + 1);
            const tomorrowDate = tomorrow.toISOString().split('T')[0];

            const apiPath = this.demoMode ? '/api/demo/suggestions/multiple' : '/api/suggestions/multiple';
            console.log(`Debug: Calling Suggestions API: ${apiPath} (demoMode: ${this.demoMode})`);
            const response = await fetch(apiPath, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    user_id: this.currentUserId,
                    date: tomorrowDate
                })
            });

            const data = await response.json();
            
            if (data.status === 'success') {
                this.renderSuggestions(data.suggestions);
            } else {
                throw new Error(data.message || '提案取得エラー');
            }
        } catch (error) {
            console.error('提案取得エラー:', error);
            this.showSuggestionsError();
        }
    }

    async loadDailyAdvice() {
        // 現在の時刻に基づいてアドバイスを生成
        const currentHour = new Date().getHours();
        const advice = this.generateAdvice(currentHour);
        this.updateAdviceDisplay(advice);
    }

    updateWorkloadDisplay(workloadData) {
        // 負荷メーターの更新
        const workloadNumber = document.getElementById('workload-number');
        const avgStress = document.getElementById('avg-stress');
        const totalWorkload = document.getElementById('total-workload');
        const highStressHours = document.getElementById('high-stress-hours');

        if (workloadNumber) {
            const workloadValue = Math.round(workloadData.total_workload || 0);
            workloadNumber.textContent = workloadValue;
            workloadNumber.classList.add('flash-update');
            setTimeout(() => workloadNumber.classList.remove('flash-update'), 300);
        }

        if (avgStress) {
            avgStress.textContent = Math.round(workloadData.avg_stress || 0);
        }

        if (totalWorkload) {
            totalWorkload.textContent = Math.round(workloadData.total_workload || 0);
        }

        if (highStressHours) {
            highStressHours.textContent = Math.round(workloadData.high_stress_hours || 0);
        }

        // 負荷メーターの円グラフ更新
        this.updateWorkloadMeter(workloadData.total_workload || 0);
    }

    updateWorkloadMeter(workloadValue) {
        const workloadMeter = document.getElementById('workload-meter');
        if (!workloadMeter) return;

        // 負荷値を0-1000の範囲でパーセンテージに変換
        const percentage = Math.min(workloadValue / 1000 * 100, 100);
        const rotation = percentage * 3.6; // 360度の何％か

        let color;
        if (workloadValue < 400) {
            color = '#10b981'; // 緑
        } else if (workloadValue < 700) {
            color = '#f59e0b'; // 黄
        } else {
            color = '#ef4444'; // 赤
        }

        workloadMeter.style.background = `conic-gradient(
            ${color} 0deg ${rotation}deg,
            rgba(255, 255, 255, 0.1) ${rotation}deg 360deg
        )`;
    }

    updateNASAChart(workloadData) {
        const nasaData = workloadData.nasa_averages || {};
        const maxNasaItem = document.getElementById('max-nasa-item');
        
        if (maxNasaItem) {
            maxNasaItem.textContent = workloadData.max_nasa_item || '--';
        }
        
        if (Object.keys(nasaData).length === 0) return;

        const ctx = document.getElementById('nasa-chart');
        if (!ctx) return;

        // 既存のチャートを破棄
        if (this.nasaChart) {
            this.nasaChart.destroy();
        }

        // NASA項目のラベルを日本語に変換
        const labelMap = {
            'NASA_M': '精神的要求',
            'NASA_P': '身体的要求',
            'NASA_T': '時間的切迫感',
            'NASA_O': '達成度',
            'NASA_E': '努力',
            'NASA_F': 'フラストレーション'
        };

        const labels = Object.keys(nasaData).map(key => labelMap[key] || key);
        const values = Object.values(nasaData);

        // 新しいチャートを作成
        this.nasaChart = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'NASA-TLX',
                    data: values,
                    backgroundColor: 'rgba(96, 165, 250, 0.2)',
                    borderColor: 'rgba(96, 165, 250, 1)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(96, 165, 250, 1)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgba(96, 165, 250, 1)'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    r: {
                        min: 0,
                        max: 100,
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.7)',
                            stepSize: 25,
                            display: false
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.2)'
                        },
                        pointLabels: {
                            color: 'rgba(255, 255, 255, 0.9)',
                            font: {
                                size: 11
                            }
                        }
                    }
                }
            }
        });
    }

    updateWellnessMetrics(workloadData) {
        this.updateMetric('stress-tolerance', 'tolerance-bar', workloadData.stress_tolerance || 0);
        this.updateMetric('activity-balance', 'balance-bar', workloadData.activity_balance || 0);
        this.updateMetric('recovery-efficiency', 'recovery-bar', workloadData.recovery_efficiency || 0);
    }

    updateMetric(valueId, barId, value) {
        const valueElement = document.getElementById(valueId);
        const barElement = document.getElementById(barId);

        if (valueElement) {
            valueElement.textContent = `${Math.round(value)}%`;
            valueElement.classList.add('flash-update');
            setTimeout(() => valueElement.classList.remove('flash-update'), 300);
        }

        if (barElement) {
            barElement.style.width = `${Math.min(value, 100)}%`;
        }
    }

    renderOptimalPattern(timeline) {
        const container = document.getElementById('today-optimal-timeline');
        if (!container) return;

        container.innerHTML = '';

        // 連続する同じ活動をまとめて表示（24時間連続表示）
        const groupedActivities = this.groupContinuousActivities(timeline);
        
        // 24時間を1行で表示
        const timelineRow = document.createElement('div');
        timelineRow.className = 'timeline-row full-day-row';
        this.renderTimelineRow(timelineRow, groupedActivities, '');
        container.appendChild(timelineRow);

        container.classList.add('fade-in');
    }

    renderTimelineRow(rowContainer, activities, timeLabel) {
        // Flexboxベースの24時間タイムライン
        const blocksContainer = document.createElement('div');
        blocksContainer.className = 'timeline-axis-container';
        
        // 各活動を正確な時間位置とサイズで配置
        activities.forEach((group) => {
            const startHour = this.parseTimeToHour(group.startTime);
            const endHour = this.parseTimeToHour(group.endTime);
            const duration = endHour > startHour ? endHour - startHour : (24 - startHour) + endHour;
            
            const activityElement = document.createElement('div');
            activityElement.className = 'activity-block';
            
            // 絶対位置で正確な位置と幅を設定
            const leftPercentage = (startHour / 24) * 100;
            const widthPercentage = Math.max((duration / 24) * 100, 3); // 最小幅3%を保証
            activityElement.style.position = 'absolute';
            activityElement.style.left = `${leftPercentage}%`;
            activityElement.style.width = `${widthPercentage}%`;
            activityElement.style.top = '50%';
            activityElement.style.transform = 'translateY(-50%)';
            activityElement.style.height = '60px';
            
            // 活動名と時間を表示
            activityElement.innerHTML = `
                <div class="activity-name">${group.optimal_activity}</div>
                <div class="activity-time-label">${group.startTime}-${group.endTime}</div>
            `;

            // 改善効果に基づいてクラスを設定
            if (!group.was_changeable) {
                activityElement.classList.add('fixed-plan');
            } else if (group.avgImprovement > 20) {
                activityElement.classList.add('high-improvement');
            } else if (group.avgImprovement > 10) {
                activityElement.classList.add('medium-improvement');
            } else if (group.avgImprovement > 0) {
                activityElement.classList.add('low-improvement');
            } else {
                activityElement.classList.add('optimal');
            }

            // ツールチップ情報
            const duration_minutes = group.segments.length * 15;
            activityElement.title = `【${group.startTime}-${group.endTime}】\n${group.optimal_activity}\n改善効果: ${group.avgImprovement}ポイント\n継続時間: ${this.formatDuration(duration_minutes)}分`;

            blocksContainer.appendChild(activityElement);
        });

        // 時間数値を表示（3時間ごと）
        for (let hour = 0; hour < 24; hour += 3) {
            const numberElement = document.createElement('div');
            numberElement.className = 'timeline-number';
            numberElement.style.left = `${(hour / 24) * 100}%`;
            numberElement.textContent = hour.toString();
            blocksContainer.appendChild(numberElement);
        }

        rowContainer.appendChild(blocksContainer);
    }

    aggregateTimelineByHour(timeline) {
        const hourlyData = [];
        
        // 15分間隔データ（96セグメント）を時間単位にまとめる
        for (let hour = 0; hour < 24; hour++) {
            // その時間の15分間隔データを取得
            const hourSegments = timeline.filter(segment => segment.hour === hour);
            
            if (hourSegments.length > 0) {
                // 代表的なデータを選択（最初のセグメント）
                const firstSegment = hourSegments[0];
                
                // 改善効果の平均を計算
                const avgImprovement = hourSegments.reduce((sum, seg) => sum + seg.improvement, 0) / hourSegments.length;
                
                const hourData = {
                    hour: hour,
                    actual_activity: firstSegment.actual_activity,
                    optimal_activity: firstSegment.optimal_activity,
                    improvement: Math.round(avgImprovement),
                    was_changeable: firstSegment.was_changeable,
                    analysis: firstSegment.analysis,
                    quarterActivities: hourSegments  // 15分詳細データを保持
                };
                
                hourlyData.push(hourData);
            }
        }
        
        return hourlyData;
    }

    groupContinuousActivities(timeline) {
        const groups = [];
        let currentGroup = null;
        
        timeline.forEach((segment) => {
            const activityKey = `${segment.optimal_activity}_${segment.was_changeable}`;
            
            if (!currentGroup || currentGroup.activityKey !== activityKey) {
                // 新しいグループを開始
                if (currentGroup) {
                    groups.push(this.finalizeGroup(currentGroup));
                }
                
                currentGroup = {
                    activityKey: activityKey,
                    optimal_activity: segment.optimal_activity,
                    actual_activity: segment.actual_activity,
                    was_changeable: segment.was_changeable,
                    segments: [segment],
                    startTime: segment.time_label,
                    totalImprovement: segment.improvement
                };
            } else {
                // 既存のグループに追加
                currentGroup.segments.push(segment);
                currentGroup.totalImprovement += segment.improvement;
            }
        });
        
        // 最後のグループを追加
        if (currentGroup) {
            groups.push(this.finalizeGroup(currentGroup));
        }
        
        return groups;
    }

    finalizeGroup(group) {
        const lastSegment = group.segments[group.segments.length - 1];
        const duration = group.segments.length * 15; // 分単位
        
        // 最後のセグメントの時刻から終了時間を計算
        const [endHourStr, endMinuteStr] = lastSegment.time_label.split(':');
        const endHour = parseInt(endHourStr);
        const endMinute = parseInt(endMinuteStr) + 15;
        const adjustedEndHour = endMinute >= 60 ? endHour + 1 : endHour;
        const adjustedEndMinute = endMinute >= 60 ? endMinute - 60 : endMinute;
        
        // 24時間を超える場合の処理
        const finalEndHour = adjustedEndHour >= 24 ? adjustedEndHour - 24 : adjustedEndHour;
        
        return {
            ...group,
            endTime: `${finalEndHour.toString().padStart(2, '0')}:${adjustedEndMinute.toString().padStart(2, '0')}`,
            avgImprovement: Math.round(group.totalImprovement / group.segments.length),
            duration: duration
        };
    }

    formatDuration(minutes) {
        if (minutes < 60) {
            return minutes.toString();
        } else {
            const hours = Math.floor(minutes / 60);
            const remainingMinutes = minutes % 60;
            if (remainingMinutes === 0) {
                return `${hours}時間`;
            } else {
                return `${hours}時間${remainingMinutes}`;
            }
        }
    }

    groupContinuousSuggestionActivities(timeline) {
        if (!timeline || timeline.length === 0) return [];
        
        const groups = [];
        let currentGroup = null;
        
        timeline.forEach((segment) => {
            const shouldGroupWithCurrent = currentGroup && 
                currentGroup.activity === segment.activity &&
                currentGroup.changeable === segment.changeable;
            
            if (shouldGroupWithCurrent) {
                // 現在のグループに追加
                currentGroup.segments.push(segment);
                currentGroup.totalStress += segment.stress_level || 0;
                currentGroup.endTime = segment.time_label;
                
                // 理由をまとめる（重複を避ける）
                if (segment.reason && !currentGroup.reasons.includes(segment.reason)) {
                    currentGroup.reasons.push(segment.reason);
                }
            } else {
                // 前のグループを完了
                if (currentGroup) {
                    this.finalizeSuggestionGroup(currentGroup);
                    groups.push(currentGroup);
                }
                
                // 新しいグループを開始
                currentGroup = {
                    activity: segment.activity,
                    changeable: segment.changeable,
                    segments: [segment],
                    totalStress: segment.stress_level || 0,
                    startTime: segment.time_label,
                    endTime: segment.time_label,
                    reasons: segment.reason ? [segment.reason] : []
                };
            }
        });
        
        // 最後のグループを完了
        if (currentGroup) {
            this.finalizeSuggestionGroup(currentGroup);
            groups.push(currentGroup);
        }
        
        return groups;
    }

    finalizeSuggestionGroup(group) {
        // 平均ストレス計算
        group.avgStress = group.segments.length > 0 ? 
            Math.round(group.totalStress / group.segments.length) : 0;
        
        // 終了時間を正しく計算（最後のセグメント + 15分）
        if (group.segments.length > 0) {
            const lastSegment = group.segments[group.segments.length - 1];
            const [endHourStr, endMinuteStr] = lastSegment.time_label.split(':');
            const endHour = parseInt(endHourStr);
            const endMinute = parseInt(endMinuteStr) + 15;
            const adjustedEndHour = endMinute >= 60 ? endHour + 1 : endHour;
            const adjustedEndMinute = endMinute >= 60 ? endMinute - 60 : endMinute;
            
            // 24時間を超える場合の処理
            const finalEndHour = adjustedEndHour >= 24 ? adjustedEndHour - 24 : adjustedEndHour;
            
            group.endTime = `${finalEndHour.toString().padStart(2, '0')}:${adjustedEndMinute.toString().padStart(2, '0')}`;
        }
        
        // 理由を結合
        group.reason = group.reasons.length > 0 ? 
            group.reasons.join('、') : 'ストレス軽減効果';
    }

    updatePatternSummary(summary) {
        const summaryElement = document.getElementById('today-pattern-summary');
        if (!summaryElement || !summary) return;

        // サマリー情報を表示
        const summaryHtml = `
            <div class="summary-stats">
                <div class="stat-item">
                    <span class="stat-label">改善可能性:</span>
                    <span class="stat-value">${summary.total_stress_reduction_potential}ポイント</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">要改善時間:</span>
                    <span class="stat-value">${summary.high_improvement_opportunities}時間</span>
                </div>
            </div>
            <div class="key-insights">
                <h4>主な気づき:</h4>
                <ul>
                    ${summary.key_insights.map(insight => `<li>${insight}</li>`).join('')}
                </ul>
            </div>
            <div class="overall-message">
                ${summary.overall_message}
            </div>
        `;

        summaryElement.innerHTML = summaryHtml;
    }

    renderSuggestions(suggestions) {
        if (!suggestions || suggestions.length < 2) return;

        // 提案データを保存
        this.suggestions[0] = suggestions[0];
        this.suggestions[1] = suggestions[1];

        // 現在選択中の提案を表示
        this.updateCurrentSuggestionDisplay();
    }

    renderSingleSuggestion(suggestion, timelineId, reductionId, changesId) {
        // タイムライン表示
        const timelineContainer = document.getElementById(timelineId);
        if (timelineContainer && suggestion.timeline) {
            timelineContainer.innerHTML = '';

            // 連続する同じ活動をまとめて表示（2行構成）
            const groupedActivities = this.groupContinuousSuggestionActivities(suggestion.timeline);
            
            // 0:00-12:00（前半）と12:00-24:00（後半）に分割
            const morningActivities = groupedActivities.filter(group => {
                const startHour = parseInt(group.startTime.split(':')[0]);
                return startHour < 12;
            });
            
            const afternoonActivities = groupedActivities.filter(group => {
                const startHour = parseInt(group.startTime.split(':')[0]);
                return startHour >= 12;
            });

            // 前半行（0:00-12:00）を表示
            const morningRow = document.createElement('div');
            morningRow.className = 'timeline-row morning-row';
            this.renderSuggestionTimelineRow(morningRow, morningActivities, '');
            timelineContainer.appendChild(morningRow);

            // 後半行（12:00-24:00）を表示
            const afternoonRow = document.createElement('div');
            afternoonRow.className = 'timeline-row afternoon-row';
            this.renderSuggestionTimelineRow(afternoonRow, afternoonActivities, '');
            timelineContainer.appendChild(afternoonRow);

            timelineContainer.classList.add('fade-in');
        }
    }

    renderSuggestionTimelineRow(rowContainer, activities, timeLabel) {
        // Flexboxベースの24時間タイムライン
        const timelineAxisContainer = document.createElement('div');
        timelineAxisContainer.className = 'timeline-axis-container';
        
        // 各活動を正確な時間位置とサイズで配置
        activities.forEach((group) => {
            const startHour = this.parseTimeToHour(group.startTime);
            const endHour = this.parseTimeToHour(group.endTime);
            const duration = endHour > startHour ? endHour - startHour : (24 - startHour) + endHour;
            
            const activityElement = document.createElement('div');
            activityElement.className = 'activity-block';
            
            // 絶対位置で正確な位置と幅を設定
            const leftPercentage = (startHour / 24) * 100;
            const widthPercentage = Math.max((duration / 24) * 100, 3); // 最小幅3%を保証
            activityElement.style.position = 'absolute';
            activityElement.style.left = `${leftPercentage}%`;
            activityElement.style.width = `${widthPercentage}%`;
            activityElement.style.top = '50%';
            activityElement.style.transform = 'translateY(-50%)';
            activityElement.style.height = '60px';
            
            // 活動名と時間を表示
            activityElement.innerHTML = `
                <div class="activity-name">${group.activity}</div>
                <div class="activity-time-label">${group.startTime}-${group.endTime}</div>
            `;

            // ストレスレベルに基づいてクラスを設定
            if (!group.changeable) {
                activityElement.classList.add('fixed-plan');
            } else if (group.avgStress <= 30) {
                activityElement.classList.add('low-stress');
            } else if (group.avgStress <= 60) {
                activityElement.classList.add('medium-stress');
            } else {
                activityElement.classList.add('high-stress');
            }

            // ツールチップ情報
            const duration_minutes = group.segments.length * 15;
            activityElement.title = `【${group.startTime}-${group.endTime}】\n${group.activity}\nストレス: ${group.avgStress}\n継続時間: ${this.formatDuration(duration_minutes)}分`;

            timelineAxisContainer.appendChild(activityElement);
        });
        // 時間数値を表示（3時間ごと）
        for (let hour = 0; hour < 24; hour += 3) {
            const numberElement = document.createElement('div');
            numberElement.className = 'timeline-number';
            numberElement.style.left = `${(hour / 24) * 100}%`;
            numberElement.textContent = hour.toString();
            timelineAxisContainer.appendChild(numberElement);
        }
        
        rowContainer.appendChild(timelineAxisContainer);
    }

    aggregateSuggestionTimelineByHour(timeline) {
        const hourlyData = [];
        
        // 15分間隔データ（96セグメント）を時間単位にまとめる
        for (let hour = 0; hour < 24; hour++) {
            // その時間の15分間隔データを取得
            const hourSegments = timeline.filter(segment => segment.hour === hour);
            
            if (hourSegments.length > 0) {
                // 代表的なデータを選択（最初のセグメント）
                const firstSegment = hourSegments[0];
                
                // ストレスレベルの平均を計算
                const avgStress = Math.round(hourSegments.reduce((sum, seg) => sum + seg.stress_level, 0) / hourSegments.length);
                
                const hourData = {
                    hour: hour,
                    activity: firstSegment.activity,
                    stress_level: avgStress,
                    changeable: firstSegment.changeable,
                    reason: firstSegment.reason,
                    quarterActivities: hourSegments  // 15分詳細データを保持
                };
                
                hourlyData.push(hourData);
            }
        }
        
        return hourlyData;
    }

    generateAdvice(currentHour) {
        const timeBasedAdvice = {
            morning: [
                "おはようございます。深呼吸をして一日をスタートしましょう。",
                "朝の軽い運動は一日のエネルギーレベルを向上させます。",
                "水分補給を忘れずに。脳の機能を最適化します。"
            ],
            afternoon: [
                "午後のエネルギー低下は自然なことです。短い休憩を取りましょう。",
                "軽いストレッチで血流を改善し、集中力を回復させましょう。",
                "適度な間食で血糖値を安定させることができます。"
            ],
            evening: [
                "お疲れ様でした。今日の成果を振り返ってみましょう。",
                "夕方以降はブルーライトを控えめにして、睡眠の質を向上させましょう。",
                "リラクゼーション技法で一日の緊張を和らげましょう。"
            ],
            night: [
                "就寝前のルーティンで心と体を休息モードに切り替えましょう。",
                "明日のタスクを軽く整理して、心配事を減らしましょう。",
                "質の良い睡眠が明日のパフォーマンスを決定します。"
            ]
        };

        let advice;
        if (currentHour >= 5 && currentHour < 12) {
            advice = timeBasedAdvice.morning;
        } else if (currentHour >= 12 && currentHour < 17) {
            advice = timeBasedAdvice.afternoon;
        } else if (currentHour >= 17 && currentHour < 21) {
            advice = timeBasedAdvice.evening;
        } else {
            advice = timeBasedAdvice.night;
        }

        return advice[Math.floor(Math.random() * advice.length)];
    }

    updateAdviceDisplay(advice) {
        const adviceElement = document.getElementById('daily-advice');
        if (adviceElement) {
            adviceElement.textContent = advice;
            adviceElement.classList.add('fade-in');
        }
    }

    showSaveIndicator() {
        const indicator = document.getElementById('save-indicator');
        if (indicator) {
            indicator.classList.add('show');
            setTimeout(() => {
                indicator.classList.remove('show');
            }, 3000);
        }
    }

    updateConnectionStatus(status) {
        const statusDot = document.getElementById('connection-status');
        const statusText = document.getElementById('status-text');

        if (!statusDot || !statusText) return;

        switch (status) {
            case 'connected':
                statusDot.className = 'status-dot';
                statusText.textContent = '正常';
                break;
            case 'connecting':
                statusDot.className = 'status-dot warning';
                statusText.textContent = '接続中';
                break;
            case 'error':
                statusDot.className = 'status-dot error';
                statusText.textContent = 'エラー';
                break;
        }
    }

    updateLastUpdateTime() {
        this.lastUpdateTime = new Date();
        const lastUpdateElement = document.getElementById('last-update-time');
        if (lastUpdateElement) {
            lastUpdateElement.textContent = this.lastUpdateTime.toLocaleTimeString('ja-JP', {
                hour: '2-digit',
                minute: '2-digit'
            });
        }
    }

    showWorkloadError() {
        const workloadNumber = document.getElementById('workload-number');
        if (workloadNumber) {
            workloadNumber.textContent = '--';
        }
    }

    showPatternError() {
        const container = document.getElementById('today-optimal-timeline');
        if (container) {
            container.innerHTML = '<div class="loading">データ取得エラー</div>';
        }
    }

    showSuggestionsError() {
        const containerA = document.getElementById('suggestion-a-timeline');
        const containerB = document.getElementById('suggestion-b-timeline');
        
        if (containerA) {
            containerA.innerHTML = '<div class="loading">エラー</div>';
        }
        if (containerB) {
            containerB.innerHTML = '<div class="loading">エラー</div>';
        }
    }

    setupUserSelector() {
        const userSelectorBtn = document.getElementById('user-selector-btn');
        const userDropdown = document.getElementById('user-dropdown');
        const currentUserName = document.getElementById('current-user-name');

        // ユーザーアイコンクリックイベント
        userSelectorBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            userDropdown.classList.toggle('show');
        });

        // ドロップダウン外クリックで閉じる
        document.addEventListener('click', (e) => {
            if (!userDropdown.contains(e.target) && !userSelectorBtn.contains(e.target)) {
                userDropdown.classList.remove('show');
            }
        });

        // ユーザー選択イベント
        const dropdownItems = userDropdown.querySelectorAll('.dropdown-item');
        dropdownItems.forEach(item => {
            item.addEventListener('click', (e) => {
                const selectedUser = item.getAttribute('data-user');
                const userName = item.querySelector('span').textContent;
                
                // アクティブ状態を更新
                dropdownItems.forEach(i => i.classList.remove('active'));
                item.classList.add('active');
                
                // ユーザー切り替え
                this.switchUser(selectedUser, userName);
                
                // ドロップダウンを閉じる
                userDropdown.classList.remove('show');
            });
        });

        // 初期選択状態を設定
        const defaultItem = userDropdown.querySelector('[data-user="default"]');
        if (defaultItem) {
            defaultItem.classList.add('active');
        }
    }

    switchUser(userId, userName) {
        console.log(`ユーザー切り替え: ${userId} (${userName})`);
        
        // 現在のユーザーを更新
        this.currentUserId = userId;
        const currentUserName = document.getElementById('current-user-name');
        if (currentUserName) {
            currentUserName.textContent = userName;
        }
        
        // データを再読み込み
        this.loadAllData();
    }

    // 提案切り替えコントロールのセットアップ
    setupSuggestionControls() {
        const manualSwitchBtn = document.getElementById('manual-switch-btn');
        const voiceSwitchBtn = document.getElementById('voice-switch-btn');
        
        if (manualSwitchBtn) {
            manualSwitchBtn.addEventListener('click', () => {
                this.switchSuggestion();
            });
        }
        
        if (voiceSwitchBtn) {
            voiceSwitchBtn.addEventListener('click', () => {
                this.toggleVoiceRecognition();
            });
        }
    }

    // 音声認識のセットアップ
    setupVoiceRecognition() {
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            this.recognition = new SpeechRecognition();
            this.recognition.lang = 'ja-JP';
            this.recognition.continuous = false;
            this.recognition.interimResults = false;

            this.recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript.toLowerCase();
                console.log('音声認識結果:', transcript);
                
                if (transcript.includes('切り替え') || transcript.includes('チェンジ') || 
                    transcript.includes('別の') || transcript.includes('次')) {
                    this.switchSuggestion();
                }
            };

            this.recognition.onerror = (event) => {
                console.error('音声認識エラー:', event.error);
                this.stopVoiceRecognition();
            };

            this.recognition.onend = () => {
                if (this.isListening) {
                    // 連続的に音声認識を継続
                    setTimeout(() => {
                        if (this.isListening) {
                            this.recognition.start();
                        }
                    }, 100);
                }
            };
        }
    }

    // 音声認識の開始/停止切り替え
    toggleVoiceRecognition() {
        if (!this.recognition) {
            console.warn('音声認識はサポートされていません');
            return;
        }

        if (this.isListening) {
            this.stopVoiceRecognition();
        } else {
            this.startVoiceRecognition();
        }
    }

    // 音声認識開始
    startVoiceRecognition() {
        if (!this.recognition) return;

        this.isListening = true;
        this.recognition.start();
        
        const voiceBtn = document.getElementById('voice-switch-btn');
        if (voiceBtn) {
            voiceBtn.classList.add('active');
        }
        
        console.log('音声認識開始 - "切り替え"と言ってください');
    }

    // 音声認識停止
    stopVoiceRecognition() {
        if (!this.recognition) return;

        this.isListening = false;
        this.recognition.stop();
        
        const voiceBtn = document.getElementById('voice-switch-btn');
        if (voiceBtn) {
            voiceBtn.classList.remove('active');
        }
        
        console.log('音声認識停止');
    }

    // 提案の切り替え
    switchSuggestion() {
        this.currentSuggestionIndex = this.currentSuggestionIndex === 0 ? 1 : 0;
        this.updateCurrentSuggestionDisplay();
        
        const indicator = document.getElementById('current-suggestion-indicator');
        if (indicator) {
            indicator.textContent = this.currentSuggestionIndex === 0 ? 'A' : 'B';
        }
        
        console.log('提案切り替え:', this.currentSuggestionIndex === 0 ? 'A' : 'B');
    }

    // 単一提案のタイムライン表示
    renderSingleSuggestionTimeline(timelineContainer, suggestion) {
        if (!timelineContainer || !suggestion.timeline) return;

        timelineContainer.innerHTML = '';

        // 連続する同じ活動をまとめて表示（24時間連続表示）
        const groupedActivities = this.groupContinuousSuggestionActivities(suggestion.timeline);
        
        // 24時間を1行で表示
        const timelineRow = document.createElement('div');
        timelineRow.className = 'timeline-row full-day-row';
        this.renderSuggestionTimelineRow(timelineRow, groupedActivities, '');
        timelineContainer.appendChild(timelineRow);

        timelineContainer.classList.add('fade-in');
    }

    // 現在の提案表示を更新
    updateCurrentSuggestionDisplay() {
        const currentSuggestion = this.suggestions[this.currentSuggestionIndex];
        if (!currentSuggestion) return;

        const title = document.getElementById('current-suggestion-title');
        const timeline = document.getElementById('current-suggestion-timeline');
        const reduction = document.getElementById('current-reduction');
        const changes = document.getElementById('current-changes');

        if (title) {
            title.textContent = `提案パターン ${this.currentSuggestionIndex === 0 ? 'A' : 'B'}`;
        }

        if (timeline) {
            timeline.innerHTML = '';
            this.renderSingleSuggestionTimeline(timeline, currentSuggestion);
        }

        if (reduction) {
            reduction.textContent = currentSuggestion.expected_reduction || '--';
        }

        if (changes) {
            changes.innerHTML = currentSuggestion.key_changes || '生成中...';
        }
    }

    // 時間文字列（HH:MM）を時間数値に変換
    parseTimeToHour(timeString) {
        const [hours, minutes] = timeString.split(':').map(Number);
        return hours + (minutes / 60);
    }
}

// アプリケーション起動
document.addEventListener('DOMContentLoaded', () => {
    new SmartMirrorStressDashboard();
});

// エラーハンドリング
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
});

// 未処理のPromise拒否をキャッチ
window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    event.preventDefault();
});