/**
 * Alpha Quantitative Engine — Dashboard Logic v2
 */

// Global Chart Instances
let sentimentBarChart = null;
let moodDoughnutChart = null;

const REFRESH_RATE_MS = 30000; // 30 seconds

// Colors (matching CSS variables)
const C_GREEN = '#00fa9a';
const C_RED = '#ff2a55';
const C_CYAN = '#00d2ff';
const C_MUTED = '#8b9bb4';

const BG_GREEN = 'rgba(0, 250, 154, 0.2)';
const BG_RED = 'rgba(255, 42, 85, 0.2)';
const BG_CYAN = 'rgba(0, 210, 255, 0.2)';

// Formatting
const fScore = (num) => {
    const n = parseFloat(num);
    return (n > 0 ? '+' : '') + n.toFixed(4);
};

// Set Global Chart Defaults
Chart.defaults.color = C_MUTED;
Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.05)';

document.addEventListener('DOMContentLoaded', () => {
    updateDashboard();
    setInterval(updateDashboard, REFRESH_RATE_MS);

    document.getElementById('refresh-btn').addEventListener('click', (e) => {
        const icon = e.currentTarget.querySelector('i');
        icon.classList.add('fa-spin');
        updateDashboard().then(() => setTimeout(() => icon.classList.remove('fa-spin'), 600));
    });
});

async function updateDashboard() {
    try {
        await Promise.all([
            fetchStats(),
            fetchOverview(),
            fetchHeadlines()
        ]);
        console.log("Terminal Sync:", new Date().toLocaleTimeString());
    } catch (err) {
        console.error("Sync Error:", err);
    }
}

// ----------------------------------------------------
// 1. Top Core Stats
// ----------------------------------------------------
async function fetchStats() {
    const res = await fetch('/api/stats');
    const data = await res.json();
    document.getElementById('stat-stocks').textContent = data.stocks_scored.toLocaleString();
    document.getElementById('stat-headlines').textContent = data.total_headlines.toLocaleString();
    // Accuracy is static in HTML since it's from training.
}

// ----------------------------------------------------
// 2. Fetch Chart Data & Movers
// ----------------------------------------------------
async function fetchOverview() {
    const res = await fetch('/api/overview');
    const data = await res.json();

    renderBarChart(data.bullish, data.bearish);
    renderMoodChart(data.bullish, data.bearish);
    renderMoversList('bullish-list', data.bullish, true);
    renderMoversList('bearish-list', data.bearish, false);
    renderTickerTape(data.bullish, data.bearish);
}

// ----------------------------------------------------
// 3. Main Bar Chart (Left Area)
// ----------------------------------------------------
function renderBarChart(bullish, bearish) {
    const ctx = document.getElementById('sentimentBarChart').getContext('2d');

    // Top 7 and Bottom 7
    const chartData = [...bullish.slice(0, 7), ...bearish.slice(0, 7)];
    chartData.sort((a, b) => b.score - a.score); // Highest to lowest

    const labels = chartData.map(d => d.ticker.replace('SECTOR_', '') + (d.ticker.includes('SECTOR') ? ' Sec' : ''));
    const scores = chartData.map(d => d.score);

    const bgColors = scores.map(s => s > 0.1 ? BG_GREEN : (s < -0.1 ? BG_RED : BG_CYAN));
    const borderColors = scores.map(s => s > 0.1 ? C_GREEN : (s < -0.1 ? C_RED : C_CYAN));

    if (sentimentBarChart) {
        sentimentBarChart.data.labels = labels;
        sentimentBarChart.data.datasets[0].data = scores;
        sentimentBarChart.data.datasets[0].backgroundColor = bgColors;
        sentimentBarChart.data.datasets[0].borderColor = borderColors;
        sentimentBarChart.update();
        return;
    }

    sentimentBarChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'AI Score',
                data: scores,
                backgroundColor: bgColors,
                borderColor: borderColors,
                borderWidth: 1.5,
                borderRadius: 4,
                barPercentage: 0.6
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(10, 14, 23, 0.95)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: 'rgba(0, 210, 255, 0.3)',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: false,
                    callbacks: { label: (ctx) => 'Score: ' + fScore(ctx.raw) }
                }
            },
            scales: {
                x: { min: -1, max: 1 },
                y: { grid: { display: false } }
            }
        }
    });
}

// ----------------------------------------------------
// 4. Market Mood Doughnut Chart
// ----------------------------------------------------
function renderMoodChart(bullish, bearish) {
    const ctx = document.getElementById('moodChart').getContext('2d');

    // Simple math to derive overall market sentiment ratio
    const bullsCount = bullish.length;
    const bearsCount = bearish.length;
    // We assume the rest are neutral out of 50 total sample
    const neutralCount = Math.max(50 - (bullsCount + bearsCount), 5);

    const total = bullsCount + bearsCount + neutralCount;
    // Calculate an arbitrary overall index score (-1 to +1 based on ratios)
    const moodIndex = ((bullsCount - bearsCount) / total).toFixed(2);

    // Update center text
    const moodLabel = document.querySelector('#mood-label .mood-val');
    moodLabel.textContent = (moodIndex > 0 ? '+' : '') + moodIndex;
    moodLabel.style.color = moodIndex > 0 ? C_GREEN : (moodIndex < 0 ? C_RED : C_CYAN);

    if (moodDoughnutChart) {
        moodDoughnutChart.data.datasets[0].data = [bullsCount, neutralCount, bearsCount];
        moodDoughnutChart.update();
        return;
    }

    moodDoughnutChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Bullish', 'Neutral', 'Bearish'],
            datasets: [{
                data: [bullsCount, neutralCount, bearsCount],
                backgroundColor: [C_GREEN, C_CYAN, C_RED],
                borderWidth: 0,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '75%',
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(0,0,0,0.8)',
                    padding: 10
                }
            }
        }
    });
}

// ----------------------------------------------------
// 5. Movers List (Bulls & Bears)
// ----------------------------------------------------
function renderMoversList(elementId, items, isBullish) {
    const container = document.getElementById(elementId);
    container.innerHTML = '';

    if (!items || items.length === 0) {
        container.innerHTML = '<div class="loading-state">Awaiting Market Data...</div>';
        return;
    }

    const colorClass = isBullish ? 'text-green' : 'text-red';

    items.slice(0, 5).forEach(item => {
        let sym = item.ticker;
        let name = item.name;
        if (sym.startsWith('SECTOR_')) {
            sym = sym.replace('SECTOR_', '');
            name = sym + ' Sector Average';
        }

        container.innerHTML += `
            <div class="mover-row" style="cursor:pointer;" onclick="openCompanyModal('${item.ticker}')">
                <div class="mover-info">
                    <span class="mover-sym">${sym}</span>
                    <span class="mover-name">${name}</span>
                </div>
                <div class="mover-score ${colorClass}">
                    ${fScore(item.score)}
                </div>
            </div>
        `;
    });
}

// ----------------------------------------------------
// 6. Live News Feed Sidebar
// ----------------------------------------------------
async function fetchHeadlines() {
    const res = await fetch('/api/headlines');
    const headlines = await res.json();

    const container = document.getElementById('news-feed');
    container.innerHTML = '';

    headlines.forEach(news => {
        const isBull = news.score > 0.2;
        const isBear = news.score < -0.2;
        const colorClass = isBull ? 'text-green' : (isBear ? 'text-red' : 'text-cyan');
        const borderClass = isBull ? 'bullish' : (isBear ? 'bearish' : '');

        let displayTicker = news.ticker.startsWith('SECTOR_') ? news.ticker.replace('SECTOR_', '') : news.ticker;

        container.innerHTML += `
            <div class="news-card ${borderClass}" style="cursor:pointer;" onclick="openCompanyModal('${news.ticker}')">
                <div class="news-top">
                    <span class="news-sym">${displayTicker}</span>
                    <span class="news-scr ${colorClass}">${fScore(news.score)}</span>
                </div>
                <div class="news-headline">${news.headline}</div>
                <div class="news-meta">
                    <span>${news.source.replace('📰', '').replace('💬', '')}</span>
                    <span>${news.time_ago}</span>
                </div>
            </div>
        `;
    });
}

// ----------------------------------------------------
// 7. Ticker Tape (Top bar)
// ----------------------------------------------------
function renderTickerTape(bulls, bears) {
    const container = document.getElementById('top-ticker');

    // Interleave bulls and bears
    const tapes = [];
    const max = Math.max(bulls.length, bears.length);
    for (let i = 0; i < max; i++) {
        if (bulls[i]) tapes.push(`<div class="ticker__item">${bulls[i].ticker.replace('SECTOR_', '')} <span class="up">▲ ${fScore(bulls[i].score)}</span></div>`);
        if (bears[i]) tapes.push(`<div class="ticker__item">${bears[i].ticker.replace('SECTOR_', '')} <span class="down">▼ ${fScore(bears[i].score)}</span></div>`);
    }

    // Duplicate to ensure smooth infinite scroll
    container.innerHTML = tapes.join('') + tapes.join('');
}

// ----------------------------------------------------
// 8. Live Search Bar
// ----------------------------------------------------
const searchInput = document.getElementById('company-search');
const searchResults = document.getElementById('search-results');
let searchTimeout;

searchInput.addEventListener('input', (e) => {
    clearTimeout(searchTimeout);
    const query = e.target.value.trim();

    if (query.length < 2) {
        searchResults.style.display = 'none';
        return;
    }

    // Debounce API calls
    searchTimeout = setTimeout(async () => {
        try {
            const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
            const data = await res.json();

            searchResults.innerHTML = '';

            if (data.results && data.results.length > 0) {
                data.results.forEach(item => {
                    const colorClass = item.score > 0.1 ? 'text-green' : (item.score < -0.1 ? 'text-red' : 'text-cyan');
                    searchResults.innerHTML += `
                        <div class="search-result-item" onclick="openCompanyModal('${item.ticker}')">
                            <div class="mover-info">
                                <span class="mover-sym">${item.ticker.replace('SECTOR_', '')}</span>
                                <span class="mover-name">${item.name}</span>
                            </div>
                            <div class="mover-score ${colorClass}">${fScore(item.score)}</div>
                        </div>
                    `;
                });
                searchResults.style.display = 'block';
            } else {
                searchResults.innerHTML = '<div class="search-result-item"><span class="text-muted">No data found</span></div>';
                searchResults.style.display = 'block';
            }
        } catch (err) {
            console.error("Search failed:", err);
        }
    }, 400);
});

// Close search on click outside
document.addEventListener('click', (e) => {
    if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {
        searchResults.style.display = 'none';
    }
});

// ----------------------------------------------------
// 9. Company Details Modal
// ----------------------------------------------------
let modalTrendChart = null;
const modal = document.getElementById('company-modal');
const closeBtn = document.getElementById('modal-close');

closeBtn.addEventListener('click', () => modal.style.display = 'none');
modal.addEventListener('click', (e) => {
    if (e.target === modal) modal.style.display = 'none';
});

async function openCompanyModal(rawTicker) {
    // Hide search if open
    document.getElementById('search-results').style.display = 'none';

    // Clean ticker
    const ticker = rawTicker.replace('SECTOR_', '');

    try {
        const res = await fetch(`/api/company/${encodeURIComponent(ticker)}`);
        const data = await res.json();

        // 1. Header Info
        document.getElementById('modal-ticker').textContent = data.ticker;
        document.getElementById('modal-name').textContent = data.name;
        document.getElementById('modal-sector').textContent = data.sector;

        const scoreEl = document.getElementById('modal-score');
        scoreEl.textContent = fScore(data.current_score);
        scoreEl.className = 'stat-val ' + (data.current_score > 0.1 ? 'text-green' : (data.current_score < -0.1 ? 'text-red' : 'text-cyan'));

        // 2. Trend Chart
        renderModalChart(data.trend);

        // 3. News List
        const newsList = document.getElementById('modal-news-list');
        newsList.innerHTML = '';
        if (data.headlines && data.headlines.length > 0) {
            data.headlines.forEach(news => {
                const isBull = news.score > 0.2;
                const isBear = news.score < -0.2;
                const colorClass = isBull ? 'text-green' : (isBear ? 'text-red' : 'text-cyan');
                const borderClass = isBull ? 'bullish' : (isBear ? 'bearish' : '');

                newsList.innerHTML += `
                    <div class="news-card ${borderClass}" style="margin-bottom:0.5rem; animation:none;">
                        <div class="news-top">
                            <span class="news-sym">${news.source.replace('📰', '')}</span>
                            <span class="news-scr ${colorClass}">${fScore(news.score)}</span>
                        </div>
                        <div class="news-headline" style="font-size:0.8rem;">${news.headline}</div>
                        <div class="news-meta">
                            <span></span>
                            <span>${news.time_ago}</span>
                        </div>
                    </div>
                `;
            });
        } else {
            newsList.innerHTML = '<div class="text-muted">No specific news found. Driven by general market sentiment.</div>';
        }

        // Show Modal
        modal.style.display = 'flex';

    } catch (err) {
        console.error("Failed to load company details", err);
    }
}

function renderModalChart(trendData) {
    const ctx = document.getElementById('modalTrendChart').getContext('2d');

    const labels = trendData.map(d => d.time_label);
    const scores = trendData.map(d => d.score);

    const isPositive = scores[scores.length - 1] >= 0;
    const lineColor = isPositive ? C_GREEN : C_RED;
    const bgColor = isPositive ? BG_GREEN : BG_RED;

    if (modalTrendChart) {
        modalTrendChart.data.labels = labels;
        modalTrendChart.data.datasets[0].data = scores;
        modalTrendChart.data.datasets[0].borderColor = lineColor;
        modalTrendChart.data.datasets[0].backgroundColor = bgColor;
        modalTrendChart.update();
        return;
    }

    modalTrendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Sentiment Trend',
                data: scores,
                borderColor: lineColor,
                backgroundColor: bgColor,
                borderWidth: 2,
                fill: true,
                tension: 0.3,
                pointRadius: 3,
                pointBackgroundColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(0,0,0,0.8)',
                    padding: 10,
                    callbacks: { label: (ctx) => 'Score: ' + fScore(ctx.raw) }
                }
            },
            scales: {
                y: { min: -1, max: 1 },
                x: { grid: { display: false } }
            }
        }
    });
}
