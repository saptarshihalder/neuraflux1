import os
import sys
import random
import time
import json
import math

def generate_sample_data(steps=100):
    """Generate rich sample training data"""
    # Training loss (decaying exponential with noise)
    train_loss = []
    loss = 5.0
    for i in range(steps):
        loss = loss * 0.95 + random.uniform(-0.2, 0.2)
        train_loss.append(max(0.1, loss))
    
    # Validation loss (slightly higher, less noise)
    val_loss = []
    loss = 5.5
    for i in range(steps):
        loss = loss * 0.96 + random.uniform(-0.1, 0.1)
        val_loss.append(max(0.2, loss))
    
    # Learning rate (step decay)
    lr = []
    current_lr = 0.001
    for i in range(steps):
        if i > 0 and i % 20 == 0:
            current_lr *= 0.5
        lr.append(current_lr)
    
    # Rewards (gradually increasing)
    rewards = []
    reward = -0.5
    for i in range(steps):
        reward = reward + 0.025 + random.uniform(-0.05, 0.05)
        rewards.append(min(reward, 1.0))  # Cap at 1.0
    
    # PPO clip fraction
    ppo_clip = []
    for i in range(steps):
        clip = 0.2 * math.exp(-0.01 * i) + random.uniform(-0.02, 0.02)
        ppo_clip.append(max(0.01, clip))
    
    # Value loss
    value_loss = []
    v_loss = 2.0
    for i in range(steps):
        v_loss = v_loss * 0.97 + random.uniform(-0.1, 0.1)
        value_loss.append(max(0.1, v_loss))
    
    # Policy KL divergence
    kl_div = []
    kl = 1.5
    for i in range(steps):
        kl = kl * 0.95 + random.uniform(-0.1, 0.1)
        kl_div.append(max(0.05, kl))
    
    # Attention map data (for later visualization)
    attention_data = []
    tokens = ["<s>", "The", "model", "performs", "well", "on", "many", "tasks", ".</s>"]
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            weight = 0.1 + 0.9 * (1.0 if i == j else 
                                 0.8 if abs(i-j) == 1 else
                                 0.3 if abs(i-j) < 3 else 
                                 random.uniform(0.01, 0.15))
            attention_data.append({
                "source": i,
                "target": j,
                "weight": weight
            })
    
    return {
        "steps": list(range(steps)),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "learning_rate": lr,
        "rewards": rewards,
        "ppo_clip_fraction": ppo_clip,
        "value_loss": value_loss,
        "kl_divergence": kl_div,
        "attention": {
            "tokens": tokens,
            "weights": attention_data
        }
    }

def create_enhanced_visualization():
    """Create enhanced visualizations without external libraries"""
    print("\n===== CREATING ENHANCED VISUALIZATIONS =====")
    
    # Create output directory
    os.makedirs("visualizations", exist_ok=True)
    
    # Generate sample data
    data = generate_sample_data(100)
    
    # Save data to JSON for the interactive visualization
    with open("visualizations/training_data.json", "w") as f:
        json.dump(data, f, indent=2)
    
    # Create an enhanced HTML visualization with interactive features
    with open("visualizations/enhanced_dashboard.html", "w") as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
    <title>NeuraFlux Training Dashboard</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f9f9f9;
            color: #333;
        }
        .dashboard {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 8px 8px 0 0;
            margin-bottom: 20px;
        }
        .tab-container {
            display: flex;
            background-color: #34495e;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 20px;
        }
        .tab {
            padding: 12px 20px;
            cursor: pointer;
            color: white;
            font-weight: bold;
            transition: background-color 0.3s;
        }
        .tab:hover {
            background-color: #2c3e50;
        }
        .tab.active {
            background-color: #2980b9;
        }
        .content {
            display: none;
        }
        .content.active {
            display: block;
        }
        .chart-container {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
        }
        .chart-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
        }
        .chart {
            height: 300px;
            position: relative;
        }
        .chart-title {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #2c3e50;
        }
        .tooltip {
            position: absolute;
            padding: 8px;
            background: rgba(0,0,0,0.7);
            color: white;
            border-radius: 4px;
            pointer-events: none;
            z-index: 100;
            font-size: 12px;
            display: none;
        }
        .attention-heatmap {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(30px, 1fr));
            gap: 2px;
            margin-top: 30px;
        }
        .heatmap-cell {
            width: 30px;
            height: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            color: white;
            transition: transform 0.2s;
        }
        .heatmap-cell:hover {
            transform: scale(1.2);
            z-index: 10;
        }
        .token-label {
            writing-mode: vertical-rl;
            text-orientation: mixed;
            padding: 5px;
            font-size: 10px;
        }
        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            align-items: center;
        }
        button, select {
            padding: 8px 12px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        button:hover {
            background-color: #2980b9;
        }
        .slider-container {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 20px;
        }
        input[type="range"] {
            width: 300px;
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>NeuraFlux Training Dashboard</h1>
            <p>Generated: """ + time.strftime('%Y-%m-%d %H:%M:%S') + """</p>
        </div>
        
        <div class="tab-container">
            <div class="tab active" data-tab="training-metrics">Training Metrics</div>
            <div class="tab" data-tab="rl-metrics">RL Metrics</div>
            <div class="tab" data-tab="attention">Attention Visualization</div>
        </div>
        
        <div id="training-metrics" class="content active">
            <div class="controls">
                <button id="smooth-toggle">Toggle Smoothing</button>
                <button id="download-csv">Download CSV</button>
                <div class="slider-container">
                    <span>Zoom:</span>
                    <input type="range" id="zoom-slider" min="10" max="100" value="100">
                    <span id="zoom-value">100%</span>
                </div>
            </div>
            
            <div class="chart-grid">
                <div class="chart-container">
                    <div class="chart-title">Training Loss vs Validation Loss</div>
                    <div class="chart" id="loss-chart"></div>
                </div>
                <div class="chart-container">
                    <div class="chart-title">Learning Rate</div>
                    <div class="chart" id="lr-chart"></div>
                </div>
            </div>
        </div>
        
        <div id="rl-metrics" class="content">
            <div class="controls">
                <button id="smooth-toggle-rl">Toggle Smoothing</button>
                <select id="rl-metric-select">
                    <option value="all">All Metrics</option>
                    <option value="rewards">Rewards Only</option>
                    <option value="ppo">PPO Metrics Only</option>
                </select>
            </div>
            
            <div class="chart-grid">
                <div class="chart-container">
                    <div class="chart-title">Rewards</div>
                    <div class="chart" id="rewards-chart"></div>
                </div>
                <div class="chart-container">
                    <div class="chart-title">PPO Clip Fraction</div>
                    <div class="chart" id="ppo-clip-chart"></div>
                </div>
                <div class="chart-container">
                    <div class="chart-title">Value Loss</div>
                    <div class="chart" id="value-loss-chart"></div>
                </div>
                <div class="chart-container">
                    <div class="chart-title">KL Divergence</div>
                    <div class="chart" id="kl-chart"></div>
                </div>
            </div>
        </div>
        
        <div id="attention" class="content">
            <div class="controls">
                <span>Select Head:</span>
                <select id="head-select">
                    <option value="0">Head 1</option>
                    <option value="1">Head 2</option>
                    <option value="2">Head 3</option>
                </select>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">Attention Weights</div>
                <div id="attention-heatmap"></div>
            </div>
        </div>
    </div>
    
    <div class="tooltip" id="tooltip"></div>
    
    <script>
        // Load the data
        fetch('training_data.json')
            .then(response => response.json())
            .then(data => {
                // Store data
                window.trainingData = data;
                
                // Initialize charts
                initializeCharts(data);
                initializeAttentionMap(data.attention);
                
                // Set up event listeners
                setupEventListeners();
            })
            .catch(error => {
                console.error('Error loading data:', error);
                document.body.innerHTML = '<div style="padding: 20px; color: red;">Error loading data. Please make sure training_data.json exists in the same directory.</div>';
            });
        
        // Initialize charts for training metrics
        function initializeCharts(data) {
            // Loss chart
            drawLineChart(
                'loss-chart', 
                data.steps,
                [
                    { name: 'Training Loss', values: data.train_loss, color: '#3498db' },
                    { name: 'Validation Loss', values: data.val_loss, color: '#e74c3c' }
                ],
                { yMin: 0 }
            );
            
            // Learning rate chart
            drawLineChart(
                'lr-chart',
                data.steps,
                [
                    { name: 'Learning Rate', values: data.learning_rate, color: '#9b59b6' }
                ],
                { yMin: 0 }
            );
            
            // Rewards chart
            drawLineChart(
                'rewards-chart',
                data.steps,
                [
                    { name: 'Rewards', values: data.rewards, color: '#2ecc71' }
                ]
            );
            
            // PPO clip fraction chart
            drawLineChart(
                'ppo-clip-chart',
                data.steps,
                [
                    { name: 'PPO Clip Fraction', values: data.ppo_clip_fraction, color: '#f39c12' }
                ],
                { yMin: 0 }
            );
            
            // Value loss chart
            drawLineChart(
                'value-loss-chart',
                data.steps,
                [
                    { name: 'Value Loss', values: data.value_loss, color: '#1abc9c' }
                ],
                { yMin: 0 }
            );
            
            // KL divergence chart
            drawLineChart(
                'kl-chart',
                data.steps,
                [
                    { name: 'KL Divergence', values: data.kl_divergence, color: '#e67e22' }
                ],
                { yMin: 0 }
            );
        }
        
        // Draw a line chart
        function drawLineChart(containerId, xValues, seriesList, options = {}) {
            const container = document.getElementById(containerId);
            container.innerHTML = '';
            
            // Find min/max for y-axis
            let yMin = options.yMin !== undefined ? options.yMin : Infinity;
            let yMax = -Infinity;
            
            seriesList.forEach(series => {
                series.values.forEach(value => {
                    yMin = Math.min(yMin, value);
                    yMax = Math.max(yMax, value);
                });
            });
            
            // Add some padding to y range
            const yPadding = (yMax - yMin) * 0.1;
            yMin = Math.max(0, yMin - yPadding);
            yMax = yMax + yPadding;
            
            // Create SVG
            const svgNS = "http://www.w3.org/2000/svg";
            const svg = document.createElementNS(svgNS, "svg");
            svg.setAttribute("width", "100%");
            svg.setAttribute("height", "100%");
            svg.setAttribute("viewBox", `0 0 1000 500`);
            container.appendChild(svg);
            
            // Add axes
            const axisGroup = document.createElementNS(svgNS, "g");
            axisGroup.classList.add("axes");
            
            // X-axis
            const xAxis = document.createElementNS(svgNS, "line");
            xAxis.setAttribute("x1", "50");
            xAxis.setAttribute("y1", "450");
            xAxis.setAttribute("x2", "950");
            xAxis.setAttribute("y2", "450");
            xAxis.setAttribute("stroke", "#333");
            xAxis.setAttribute("stroke-width", "2");
            axisGroup.appendChild(xAxis);
            
            // Y-axis
            const yAxis = document.createElementNS(svgNS, "line");
            yAxis.setAttribute("x1", "50");
            yAxis.setAttribute("y1", "50");
            yAxis.setAttribute("x2", "50");
            yAxis.setAttribute("y2", "450");
            yAxis.setAttribute("stroke", "#333");
            yAxis.setAttribute("stroke-width", "2");
            axisGroup.appendChild(yAxis);
            
            // X-axis labels
            const numXLabels = 5;
            for (let i = 0; i <= numXLabels; i++) {
                const xPos = 50 + (900 / numXLabels) * i;
                const xValue = Math.floor(xValues[Math.floor(xValues.length / numXLabels * i)] || 0);
                
                const xLabel = document.createElementNS(svgNS, "text");
                xLabel.setAttribute("x", xPos);
                xLabel.setAttribute("y", "470");
                xLabel.setAttribute("text-anchor", "middle");
                xLabel.setAttribute("font-size", "12");
                xLabel.textContent = xValue;
                axisGroup.appendChild(xLabel);
                
                const xTick = document.createElementNS(svgNS, "line");
                xTick.setAttribute("x1", xPos);
                xTick.setAttribute("y1", "450");
                xTick.setAttribute("x2", xPos);
                xTick.setAttribute("y2", "455");
                xTick.setAttribute("stroke", "#333");
                xTick.setAttribute("stroke-width", "2");
                axisGroup.appendChild(xTick);
            }
            
            // Y-axis labels
            const numYLabels = 5;
            for (let i = 0; i <= numYLabels; i++) {
                const yPos = 450 - (400 / numYLabels) * i;
                const yValue = (yMin + (yMax - yMin) / numYLabels * i).toFixed(2);
                
                const yLabel = document.createElementNS(svgNS, "text");
                yLabel.setAttribute("x", "40");
                yLabel.setAttribute("y", yPos + 5);
                yLabel.setAttribute("text-anchor", "end");
                yLabel.setAttribute("font-size", "12");
                yLabel.textContent = yValue;
                axisGroup.appendChild(yLabel);
                
                const yTick = document.createElementNS(svgNS, "line");
                yTick.setAttribute("x1", "45");
                yTick.setAttribute("y1", yPos);
                yTick.setAttribute("x2", "50");
                yTick.setAttribute("y2", yPos);
                yTick.setAttribute("stroke", "#333");
                yTick.setAttribute("stroke-width", "2");
                axisGroup.appendChild(yTick);
                
                // Grid line
                const gridLine = document.createElementNS(svgNS, "line");
                gridLine.setAttribute("x1", "50");
                gridLine.setAttribute("y1", yPos);
                gridLine.setAttribute("x2", "950");
                gridLine.setAttribute("y2", yPos);
                gridLine.setAttribute("stroke", "#ddd");
                gridLine.setAttribute("stroke-width", "1");
                gridLine.setAttribute("stroke-dasharray", "5,5");
                axisGroup.appendChild(gridLine);
            }
            
            svg.appendChild(axisGroup);
            
            // Legend
            const legendGroup = document.createElementNS(svgNS, "g");
            legendGroup.classList.add("legend");
            
            let legendX = 800;
            let legendY = 70;
            
            seriesList.forEach((series, index) => {
                const legendItem = document.createElementNS(svgNS, "g");
                
                const legendColor = document.createElementNS(svgNS, "rect");
                legendColor.setAttribute("x", legendX);
                legendColor.setAttribute("y", legendY + index * 25);
                legendColor.setAttribute("width", "15");
                legendColor.setAttribute("height", "15");
                legendColor.setAttribute("fill", series.color);
                legendItem.appendChild(legendColor);
                
                const legendText = document.createElementNS(svgNS, "text");
                legendText.setAttribute("x", legendX + 20);
                legendText.setAttribute("y", legendY + index * 25 + 12);
                legendText.setAttribute("font-size", "12");
                legendText.textContent = series.name;
                legendItem.appendChild(legendText);
                
                legendGroup.appendChild(legendItem);
            });
            
            svg.appendChild(legendGroup);
            
            // Plot data
            seriesList.forEach(series => {
                const lineGroup = document.createElementNS(svgNS, "g");
                lineGroup.classList.add("line-series");
                
                // Draw path
                const path = document.createElementNS(svgNS, "path");
                
                let pathData = "";
                
                xValues.forEach((x, i) => {
                    if (i < series.values.length) {
                        const x1 = mapValue(i, 0, xValues.length - 1, 50, 950);
                        const y1 = mapValue(series.values[i], yMin, yMax, 450, 50);
                        
                        if (i === 0) {
                            pathData += `M ${x1} ${y1} `;
                        } else {
                            pathData += `L ${x1} ${y1} `;
                        }
                    }
                });
                
                path.setAttribute("d", pathData);
                path.setAttribute("fill", "none");
                path.setAttribute("stroke", series.color);
                path.setAttribute("stroke-width", "2");
                lineGroup.appendChild(path);
                
                // Draw points
                xValues.forEach((x, i) => {
                    if (i < series.values.length) {
                        const x1 = mapValue(i, 0, xValues.length - 1, 50, 950);
                        const y1 = mapValue(series.values[i], yMin, yMax, 450, 50);
                        
                        const point = document.createElementNS(svgNS, "circle");
                        point.setAttribute("cx", x1);
                        point.setAttribute("cy", y1);
                        point.setAttribute("r", "3");
                        point.setAttribute("fill", series.color);
                        
                        // Add event listeners for tooltips
                        point.addEventListener("mouseover", function(e) {
                            const tooltip = document.getElementById("tooltip");
                            tooltip.style.display = "block";
                            tooltip.style.left = `${e.pageX + 10}px`;
                            tooltip.style.top = `${e.pageY - 10}px`;
                            tooltip.innerHTML = `${series.name}: ${series.values[i].toFixed(4)}<br>Step: ${x}`;
                        });
                        
                        point.addEventListener("mousemove", function(e) {
                            const tooltip = document.getElementById("tooltip");
                            tooltip.style.left = `${e.pageX + 10}px`;
                            tooltip.style.top = `${e.pageY - 10}px`;
                        });
                        
                        point.addEventListener("mouseout", function() {
                            document.getElementById("tooltip").style.display = "none";
                        });
                        
                        lineGroup.appendChild(point);
                    }
                });
                
                svg.appendChild(lineGroup);
            });
        }
        
        // Initialize attention heatmap
        function initializeAttentionMap(attentionData) {
            const container = document.getElementById('attention-heatmap');
            container.innerHTML = '';
            
            const tokens = attentionData.tokens;
            const weights = attentionData.weights;
            
            // Create token labels (column headers)
            const headerRow = document.createElement('div');
            headerRow.style.display = 'flex';
            headerRow.style.marginLeft = '30px';
            
            tokens.forEach(token => {
                const label = document.createElement('div');
                label.textContent = token;
                label.style.width = '30px';
                label.style.textAlign = 'center';
                label.style.fontSize = '10px';
                label.style.padding = '5px 0';
                headerRow.appendChild(label);
            });
            
            container.appendChild(headerRow);
            
            // Create the heatmap grid
            const grid = document.createElement('div');
            grid.style.display = 'flex';
            
            // Create row labels (source tokens)
            const rowLabels = document.createElement('div');
            
            tokens.forEach(token => {
                const label = document.createElement('div');
                label.textContent = token;
                label.className = 'token-label';
                label.style.height = '30px';
                label.style.lineHeight = '30px';
                rowLabels.appendChild(label);
            });
            
            grid.appendChild(rowLabels);
            
            // Create the actual heatmap cells
            const heatmapGrid = document.createElement('div');
            heatmapGrid.className = 'attention-heatmap';
            heatmapGrid.style.gridTemplateColumns = `repeat(${tokens.length}, 30px)`;
            
            for (let i = 0; i < tokens.length; i++) {
                for (let j = 0; j < tokens.length; j++) {
                    // Find the weight for this cell
                    const weight = weights.find(w => w.source === i && w.target === j)?.weight || 0;
                    
                    // Convert weight to color
                    const intensity = Math.min(255, Math.round(weight * 255));
                    const color = `rgb(${intensity}, ${Math.round(intensity * 0.5)}, 0)`;
                    
                    const cell = document.createElement('div');
                    cell.className = 'heatmap-cell';
                    cell.style.backgroundColor = color;
                    cell.setAttribute('data-weight', weight.toFixed(2));
                    cell.setAttribute('data-source', tokens[i]);
                    cell.setAttribute('data-target', tokens[j]);
                    
                    // Add tooltip
                    cell.addEventListener('mouseover', function(e) {
                        const tooltip = document.getElementById('tooltip');
                        tooltip.style.display = 'block';
                        tooltip.style.left = `${e.pageX + 10}px`;
                        tooltip.style.top = `${e.pageY - 10}px`;
                        tooltip.innerHTML = `${this.getAttribute('data-source')} → ${this.getAttribute('data-target')}<br>Weight: ${this.getAttribute('data-weight')}`;
                    });
                    
                    cell.addEventListener('mousemove', function(e) {
                        const tooltip = document.getElementById('tooltip');
                        tooltip.style.left = `${e.pageX + 10}px`;
                        tooltip.style.top = `${e.pageY - 10}px`;
                    });
                    
                    cell.addEventListener('mouseout', function() {
                        document.getElementById('tooltip').style.display = 'none';
                    });
                    
                    heatmapGrid.appendChild(cell);
                }
            }
            
            grid.appendChild(heatmapGrid);
            container.appendChild(grid);
        }
        
        // Helper function to map values
        function mapValue(value, inMin, inMax, outMin, outMax) {
            return (value - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
        }
        
        // Set up event listeners for interactive features
        function setupEventListeners() {
            // Tab switching
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => {
                tab.addEventListener('click', function() {
                    const tabId = this.getAttribute('data-tab');
                    
                    // Update active tab
                    tabs.forEach(t => t.classList.remove('active'));
                    this.classList.add('active');
                    
                    // Update active content
                    document.querySelectorAll('.content').forEach(content => {
                        content.classList.remove('active');
                    });
                    document.getElementById(tabId).classList.add('active');
                });
            });
            
            // Smoothing toggle
            document.getElementById('smooth-toggle').addEventListener('click', function() {
                smoothData = !smoothData;
                
                // Redraw charts with smoothed data
                const data = window.trainingData;
                
                if (smoothData) {
                    data.train_loss = smoothArray(data.train_loss, 0.8);
                    data.val_loss = smoothArray(data.val_loss, 0.8);
                    this.textContent = 'Raw Data';
                } else {
                    // Restore original data
                    location.reload();
                }
                
                initializeCharts(data);
            });
            
            // Zoom slider
            document.getElementById('zoom-slider').addEventListener('input', function() {
                const zoomValue = this.value;
                document.getElementById('zoom-value').textContent = `${zoomValue}%`;
                
                // Adjust viewBox to zoom
                const charts = document.querySelectorAll('svg');
                charts.forEach(chart => {
                    const fullViewBox = chart.getAttribute('viewBox').split(' ');
                    const newWidth = parseInt(fullViewBox[2]) * (100 / zoomValue);
                    chart.setAttribute('viewBox', `0 0 ${newWidth} ${fullViewBox[3]}`);
                });
            });
            
            // Download CSV
            document.getElementById('download-csv').addEventListener('click', function() {
                const data = window.trainingData;
                let csv = 'step,train_loss,val_loss,learning_rate,rewards,ppo_clip_fraction,value_loss,kl_divergence\\n';
                
                for (let i = 0; i < data.steps.length; i++) {
                    csv += `${data.steps[i]},`;
                    csv += `${data.train_loss[i]},`;
                    csv += `${data.val_loss[i]},`;
                    csv += `${data.learning_rate[i]},`;
                    csv += `${data.rewards[i]},`;
                    csv += `${data.ppo_clip_fraction[i]},`;
                    csv += `${data.value_loss[i]},`;
                    csv += `${data.kl_divergence[i]}\\n`;
                }
                
                const blob = new Blob([csv], { type: 'text/csv' });
                const url = URL.createObjectURL(blob);
                
                const a = document.createElement('a');
                a.setAttribute('href', url);
                a.setAttribute('download', 'training_data.csv');
                a.click();
            });
            
            // RL metric selection
            document.getElementById('rl-metric-select').addEventListener('change', function() {
                const selection = this.value;
                const rlCharts = ['rewards-chart', 'ppo-clip-chart', 'value-loss-chart', 'kl-chart'];
                
                if (selection === 'all') {
                    rlCharts.forEach(id => {
                        document.getElementById(id).parentElement.style.display = 'block';
                    });
                } else if (selection === 'rewards') {
                    rlCharts.forEach(id => {
                        document.getElementById(id).parentElement.style.display = id === 'rewards-chart' ? 'block' : 'none';
                    });
                } else if (selection === 'ppo') {
                    rlCharts.forEach(id => {
                        document.getElementById(id).parentElement.style.display = id !== 'rewards-chart' ? 'block' : 'none';
                    });
                }
            });
        }
        
        // Helper function to smooth data
        function smoothArray(arr, weight = 0.8) {
            if (arr.length === 0) return [];
            
            const smoothed = [arr[0]];
            
            for (let i = 1; i < arr.length; i++) {
                const prevSmoothed = smoothed[i-1];
                const current = arr[i];
                smoothed.push(prevSmoothed * weight + current * (1 - weight));
            }
            
            return smoothed;
        }
        
        // Smoothing flag
        let smoothData = false;
    </script>
</body>
</html>""")
    
    print(f"\n✓ Interactive dashboard created at visualizations/enhanced_dashboard.html")
    print(f"✓ Open this HTML file in a modern browser for a fully interactive experience")
    print(f"✓ Training data saved to visualizations/training_data.json")
    print("===== VISUALIZATION COMPLETE =====\n")

if __name__ == "__main__":
    try:
        print(f"Running enhanced visualization script with Python: {sys.executable}")
        create_enhanced_visualization()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        # Write error to file for debugging
        with open("error_log.txt", "w") as f:
            f.write(f"Error at {time.strftime('%Y-%m-%d %H:%M:%S')}:\n{str(e)}") 