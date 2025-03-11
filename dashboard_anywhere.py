import os
import sys
import random
import time
import shutil

def create_dashboard():
    """Create a dashboard that works from any directory"""
    # Step 1: Identify all important paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir
    
    # Navigate up until we find the project root (has the .git directory)
    max_levels = 3  # Prevent infinite loop
    for _ in range(max_levels):
        if os.path.exists(os.path.join(project_root, ".git")) or \
           os.path.exists(os.path.join(project_root, "server", "model")):
            break
        project_root = os.path.dirname(project_root)
    
    # Create visualization directories in multiple locations
    vis_dirs = [
        os.path.join(script_dir, "visualizations"),  # Local to script
        os.path.join(project_root, "visualizations")  # At project root
    ]
    
    for vis_dir in vis_dirs:
        os.makedirs(vis_dir, exist_ok=True)
        print(f"Created directory: {vis_dir}")
    
    # Generate the dashboard content
    dashboard_html = generate_dashboard_html()
    
    # Write dashboard to all locations
    dashboard_paths = []
    for vis_dir in vis_dirs:
        dashboard_path = os.path.join(vis_dir, "dashboard.html")
        with open(dashboard_path, "w", encoding="utf-8") as f:
            f.write(dashboard_html)
        dashboard_paths.append(dashboard_path)
        print(f"Created dashboard at: {dashboard_path}")
    
    # Try to open the dashboard
    try:
        if sys.platform.startswith('win'):
            os.startfile(dashboard_paths[0])
        else:
            import subprocess
            subprocess.run(["open", dashboard_paths[0]])
        print(f"✓ Dashboard opened in your browser")
    except Exception as e:
        print(f"✓ Please manually open one of these dashboards in your browser:")
        for path in dashboard_paths:
            print(f"  - {path}")
    
    # Create a batch file for easy opening
    batch_path = os.path.join(project_root, "open_dashboard.bat")
    with open(batch_path, "w") as f:
        f.write(f'start "" "{dashboard_paths[0]}"\n')
    print(f"✓ Created quick-launch batch file: {batch_path}")
    
    return dashboard_paths

def generate_dashboard_html():
    """Generate the dashboard HTML content"""
    # Generate sample data
    steps = 100
    train_loss = [5.0 * (0.97**i) + random.uniform(-0.2, 0.2) for i in range(steps)]
    val_loss = [5.5 * (0.96**i) + random.uniform(-0.1, 0.1) for i in range(steps)]
    learning_rate = [0.001 * (0.5 ** (i // 20)) for i in range(steps)]
    rewards = [-0.5 + 0.015*i + random.uniform(-0.05, 0.05) for i in range(steps)]
    
    # Create dashboard HTML with proper encoding
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>NeuraFlux Dashboard</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f0f2f5;
        }}
        .dashboard {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 8px 8px 0 0;
            margin-bottom: 20px;
        }}
        .chart-container {{
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
        }}
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
        }}
        .chart {{
            height: 300px;
            position: relative;
        }}
        .chart-title {{
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #2c3e50;
        }}
        h2 {{
            color: #2c3e50;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        .metric-card {{
            background-color: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2980b9;
            margin: 10px 0;
            opacity: 0;
            transition: opacity 0.5s;
        }}
        .metric-label {{
            font-size: 14px;
            color: #7f8c8d;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 15px;
            border-top: 1px solid #eee;
            font-size: 12px;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>NeuraFlux Training Dashboard</h1>
            <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <h2>Training Metrics</h2>
        
        <div class="chart-grid">
            <div class="chart-container">
                <div class="chart-title">Training & Validation Loss</div>
                <div class="chart">
                    <svg viewBox="0 0 800 300">
                        <!-- Axes -->
                        <line x1="50" y1="250" x2="750" y2="250" stroke="#333" stroke-width="2" />
                        <line x1="50" y1="50" x2="50" y2="250" stroke="#333" stroke-width="2" />
                        
                        <!-- Grid lines -->
                        <line x1="50" y1="150" x2="750" y2="150" stroke="#ddd" stroke-width="1" stroke-dasharray="5,5" />
                        <line x1="50" y1="100" x2="750" y2="100" stroke="#ddd" stroke-width="1" stroke-dasharray="5,5" />
                        <line x1="50" y1="200" x2="750" y2="200" stroke="#ddd" stroke-width="1" stroke-dasharray="5,5" />
                        
                        <!-- X-axis labels -->
                        <text x="50" y="270" text-anchor="middle">0</text>
                        <text x="200" y="270" text-anchor="middle">25</text>
                        <text x="400" y="270" text-anchor="middle">50</text>
                        <text x="600" y="270" text-anchor="middle">75</text>
                        <text x="750" y="270" text-anchor="middle">100</text>
                        
                        <!-- Y-axis labels -->
                        <text x="40" y="250" text-anchor="end">0</text>
                        <text x="40" y="200" text-anchor="end">1</text>
                        <text x="40" y="150" text-anchor="end">2</text>
                        <text x="40" y="100" text-anchor="end">3</text>
                        <text x="40" y="50" text-anchor="end">4</text>
                        
                        <!-- Data: Training Loss -->
                        <polyline 
                            points="{' '.join([f'{50 + i*7},{250 - max(0, min(4, val))*50}' for i, val in enumerate(train_loss)])}"
                            fill="none"
                            stroke="#3498db"
                            stroke-width="2"
                        />
                        
                        <!-- Data: Validation Loss -->
                        <polyline 
                            points="{' '.join([f'{50 + i*7},{250 - max(0, min(4, val))*50}' for i, val in enumerate(val_loss)])}"
                            fill="none"
                            stroke="#e74c3c"
                            stroke-width="2"
                            stroke-dasharray="5,5"
                        />
                        
                        <!-- Legend -->
                        <rect x="600" y="30" width="140" height="50" fill="white" stroke="#eee" />
                        <circle cx="620" y="45" r="5" fill="#3498db" />
                        <text x="630" y="50" font-size="12">Training Loss</text>
                        <circle cx="620" y="65" r="5" fill="#e74c3c" />
                        <text x="630" y="70" font-size="12">Validation Loss</text>
                    </svg>
                </div>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">Learning Rate</div>
                <div class="chart">
                    <svg viewBox="0 0 800 300">
                        <!-- Axes -->
                        <line x1="50" y1="250" x2="750" y2="250" stroke="#333" stroke-width="2" />
                        <line x1="50" y1="50" x2="50" y2="250" stroke="#333" stroke-width="2" />
                        
                        <!-- Grid lines -->
                        <line x1="50" y1="150" x2="750" y2="150" stroke="#ddd" stroke-width="1" stroke-dasharray="5,5" />
                        <line x1="50" y1="100" x2="750" y2="100" stroke="#ddd" stroke-width="1" stroke-dasharray="5,5" />
                        <line x1="50" y1="200" x2="750" y2="200" stroke="#ddd" stroke-width="1" stroke-dasharray="5,5" />
                        
                        <!-- X-axis labels -->
                        <text x="50" y="270" text-anchor="middle">0</text>
                        <text x="200" y="270" text-anchor="middle">25</text>
                        <text x="400" y="270" text-anchor="middle">50</text>
                        <text x="600" y="270" text-anchor="middle">75</text>
                        <text x="750" y="270" text-anchor="middle">100</text>
                        
                        <!-- Y-axis labels -->
                        <text x="40" y="250" text-anchor="end">0</text>
                        <text x="40" y="200" text-anchor="end">0.25</text>
                        <text x="40" y="150" text-anchor="end">0.5</text>
                        <text x="40" y="100" text-anchor="end">0.75</text>
                        <text x="40" y="50" text-anchor="end">1</text>
                        
                        <!-- Data: Learning Rate -->
                        <polyline 
                            points="{' '.join([f'{50 + i*7},{250 - val*1000}' for i, val in enumerate(learning_rate)])}"
                            fill="none"
                            stroke="#9b59b6"
                            stroke-width="2"
                        />
                        
                        <!-- Legend -->
                        <rect x="600" y="30" width="140" height="30" fill="white" stroke="#eee" />
                        <circle cx="620" y="45" r="5" fill="#9b59b6" />
                        <text x="630" y="50" font-size="12">Learning Rate</text>
                    </svg>
                </div>
            </div>
        </div>
        
        <h2>Reinforcement Learning Metrics</h2>
        
        <div class="chart-grid">
            <div class="chart-container">
                <div class="chart-title">Rewards</div>
                <div class="chart">
                    <svg viewBox="0 0 800 300">
                        <!-- Axes -->
                        <line x1="50" y1="250" x2="750" y2="250" stroke="#333" stroke-width="2" />
                        <line x1="50" y1="50" x2="50" y2="250" stroke="#333" stroke-width="2" />
                        
                        <!-- Grid lines -->
                        <line x1="50" y1="150" x2="750" y2="150" stroke="#ddd" stroke-width="1" stroke-dasharray="5,5" />
                        <line x1="50" y1="100" x2="750" y2="100" stroke="#ddd" stroke-width="1" stroke-dasharray="5,5" />
                        <line x1="50" y1="200" x2="750" y2="200" stroke="#ddd" stroke-width="1" stroke-dasharray="5,5" />
                        
                        <!-- X-axis labels -->
                        <text x="50" y="270" text-anchor="middle">0</text>
                        <text x="200" y="270" text-anchor="middle">25</text>
                        <text x="400" y="270" text-anchor="middle">50</text>
                        <text x="600" y="270" text-anchor="middle">75</text>
                        <text x="750" y="270" text-anchor="middle">100</text>
                        
                        <!-- Y-axis labels -->
                        <text x="40" y="250" text-anchor="end">-0.5</text>
                        <text x="40" y="200" text-anchor="end">0</text>
                        <text x="40" y="150" text-anchor="end">0.5</text>
                        <text x="40" y="100" text-anchor="end">1.0</text>
                        <text x="40" y="50" text-anchor="end">1.5</text>
                        
                        <!-- Data: Rewards -->
                        <polyline 
                            points="{' '.join([f'{50 + i*7},{200 - val*100}' for i, val in enumerate(rewards)])}"
                            fill="none"
                            stroke="#2ecc71"
                            stroke-width="2"
                        />
                        
                        <!-- Legend -->
                        <rect x="600" y="30" width="140" height="30" fill="white" stroke="#eee" />
                        <circle cx="620" y="45" r="5" fill="#2ecc71" />
                        <text x="630" y="50" font-size="12">Reward</text>
                    </svg>
                </div>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">Attention Heatmap</div>
                <div class="chart" style="text-align: center; padding-top: 50px;">
                    <p>Attention visualization available in the full version.</p>
                    <p>See training_chart.html for more visualizations.</p>
                </div>
            </div>
        </div>
        
        <h2>Key Metrics Summary</h2>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Final Training Loss</div>
                <div class="metric-value">{train_loss[-1]:.2f}</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Final Validation Loss</div>
                <div class="metric-value">{val_loss[-1]:.2f}</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Final Learning Rate</div>
                <div class="metric-value">{learning_rate[-1]:.6f}</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Final Reward</div>
                <div class="metric-value">{rewards[-1]:.2f}</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Training Duration</div>
                <div class="metric-value">{steps} steps</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Loss Reduction</div>
                <div class="metric-value">{(1 - train_loss[-1]/train_loss[0])*100:.1f}%</div>
            </div>
        </div>

        <div class="footer">
            <p>NeuraFlux Small Language Model - Training Visualization - {time.strftime('%Y-%m-%d')}</p>
        </div>
    </div>
    
    <script>
        // Simple animation for loading
        window.onload = function() {{
            const metrics = document.querySelectorAll('.metric-value');
            metrics.forEach(function(metric, index) {{
                setTimeout(function() {{
                    metric.style.opacity = '1';
                }}, index * 100);
            }});
        }};
    </script>
</body>
</html>"""
    
    return html_content

if __name__ == "__main__":
    print("\n===== CREATING DASHBOARD EVERYWHERE =====")
    try:
        dashboard_paths = create_dashboard()
        print("\n✓ SUCCESS! Dashboard created in multiple locations:")
        for path in dashboard_paths:
            print(f"  - {path}")
        print("\n✓ You can open the dashboard by:")
        print("  1. Double-clicking 'open_dashboard.bat' in your project root")
        print("  2. Running 'start visualizations\\dashboard.html' from any of these directories")
        print("  3. Opening the HTML file directly in your browser")
        print("\n===== DASHBOARD CREATION COMPLETE =====\n")
    except Exception as e:
        print(f"Error: {str(e)}")
        with open("dashboard_error.txt", "w") as f:
            f.write(f"Error at {time.strftime('%Y-%m-%d %H:%M:%S')}:\n{str(e)}") 