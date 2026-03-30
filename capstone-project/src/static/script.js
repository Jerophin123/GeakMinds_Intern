document.addEventListener('DOMContentLoaded', () => {
    
    // UI Sliders and displays (Row 1)
    const form = document.getElementById('prediction-form');
    const numTouchpoints = document.getElementById('num_touchpoints');
    const dspTouchpoints = document.getElementById('val_touchpoints');
    const uniqueChannels = document.getElementById('unique_channels');
    const dspChannels = document.getElementById('val_channels');
    const timeToConversion = document.getElementById('time_to_conversion_hours');
    const dspTime = document.getElementById('val_time');
    
    const firstChannel = document.getElementById('first_channel');
    const lastChannel = document.getElementById('last_channel');
    
    const probValue = document.getElementById('probability-value');
    const resultCircle = document.getElementById('result-circle');
    const predictionBadge = document.getElementById('prediction-label-badge');

    // UI Budget Allocation (Row 2)
    const budgetInput = document.getElementById('budget_input');
    const allocationGrid = document.getElementById('allocation_cards');
    const budgetInsightText = document.getElementById('budget_insight_text');

    let featureChartInstance = null;
    let fallbackTimer = null;

    // Synchronize slider text instantly
    const updateDisplays = () => {
        dspTouchpoints.textContent = numTouchpoints.value;
        dspChannels.textContent = uniqueChannels.value;
        dspTime.textContent = timeToConversion.value;
    };

    numTouchpoints.addEventListener('input', updateDisplays);
    uniqueChannels.addEventListener('input', updateDisplays);
    timeToConversion.addEventListener('input', updateDisplays);

    const formatter = new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    });

    // Fetch Attribution and render Budget Plan
    const loadBudgetAllocation = async () => {
        const totalBudget = parseFloat(budgetInput.value) || 0;
        
        try {
            const res = await fetch(`/api/budget-optimizer?total_budget=${totalBudget}`);
            const data = await res.json();
            
            if (data.error) return;

            // Render Insight Text
            if (data.summary) {
                budgetInsightText.textContent = data.summary;
            }

            // Render Cards
            allocationGrid.innerHTML = '';
            
            // Sort to show highest allocation first
            const sortedAlloc = Object.entries(data.allocations).sort((a, b) => b[1] - a[1]);

            sortedAlloc.forEach(([channel, amount]) => {
                const percentage = (data.markov_scores[channel] * 100).toFixed(1);
                
                const card = document.createElement('div');
                card.className = 'channel-card class-card-enter';
                
                // Highlight the absolute best channel in glow colors
                let nameStyle = "";
                let amountStyle = "";
                if(percentage == Math.max(...Object.values(data.markov_scores).map(v => (v*100).toFixed(1)))) {
                    nameStyle = "color: var(--accent-warning);";
                    amountStyle = "color: var(--accent-warning);";
                }

                card.innerHTML = `
                    <div class="cc-name" style="${nameStyle}">${channel}</div>
                    <div class="cc-amount" style="${amountStyle}">${formatter.format(amount)}</div>
                    <div class="cc-pct">Markov Priority: ${percentage}%</div>
                `;
                allocationGrid.appendChild(card);
            });

        } catch (e) {
            console.error("Failed to load budget optimizer", e);
        }
    };

    // Attach Budget Listener
    budgetInput.addEventListener('input', () => {
        clearTimeout(fallbackTimer);
        fallbackTimer = setTimeout(loadBudgetAllocation, 300); // 300ms debounce
    });

    // Fetch Feature Importance for Chart.js
    const loadFeatureImportance = async () => {
        try {
            const res = await fetch('/api/feature-importance');
            const data = await res.json();
            if(data.error) return;

            const labels = Object.keys(data).map(key => key.replace('first_channel_', 'First: ').replace('last_channel_', 'Last: '));
            const values = Object.values(data);

            const ctx = document.getElementById('featureChart').getContext('2d');
            Chart.defaults.color = "rgba(255, 255, 255, 0.4)";
            Chart.defaults.font.family = "'Inter', sans-serif";

            featureChartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels.slice(0, 7),
                    datasets: [{
                        label: 'Feature Weight',
                        data: values.slice(0, 7),
                        backgroundColor: 'rgba(59, 130, 246, 0.5)',
                        borderColor: 'rgba(59, 130, 246, 0.8)',
                        borderWidth: 1,
                        borderRadius: 3
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { color: "rgba(255, 255, 255, 0.05)" } },
                        y: { grid: { display: false } }
                    }
                }
            });
        } catch (e) {
            console.error(e);
        }
    };

    // Auto Inference Mechanism
    const runInference = async () => {
        resultCircle.style.transform = 'scale(0.95)';
        resultCircle.style.opacity = '0.8';

        const payload = {
            num_touchpoints: parseInt(numTouchpoints.value),
            unique_channels: parseInt(uniqueChannels.value),
            time_to_conversion_hours: parseFloat(timeToConversion.value),
            first_channel: firstChannel.value,
            last_channel: lastChannel.value
        };

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            setTimeout(() => {
                const probPercentage = (data.probability * 100).toFixed(1);
                
                probValue.textContent = `${probPercentage}%`;
                predictionBadge.textContent = data.prediction_label;

                if (data.prediction === 1) {
                    predictionBadge.className = 'status-badge converted';
                    probValue.style.color = 'var(--accent-success)';
                    resultCircle.style.borderColor = 'var(--accent-success)';
                    resultCircle.style.boxShadow = 'inset 0 0 20px rgba(16, 185, 129, 0.1), 0 0 40px rgba(16, 185, 129, 0.2)';
                } else {
                    predictionBadge.className = 'status-badge not-converted';
                    probValue.style.color = 'var(--accent-danger)';
                    resultCircle.style.borderColor = 'var(--accent-danger)';
                    resultCircle.style.boxShadow = 'inset 0 0 20px rgba(239, 68, 68, 0.1), 0 0 40px rgba(239, 68, 68, 0.2)';
                }

                resultCircle.style.transform = 'scale(1)';
                resultCircle.style.opacity = '1';

            }, 100);

        } catch (error) {
            probValue.textContent = "ERR";
            predictionBadge.textContent = "API Error";
        }
    };

    // Bind triggers
    form.addEventListener('input', runInference);
    
    // Boot up
    loadFeatureImportance();
    loadBudgetAllocation();
    runInference(); 
});
