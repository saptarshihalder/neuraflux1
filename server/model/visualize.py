import os
import sys
import random
import time

def create_visualizations():
    """Create visualizations with proper encoding"""
    print("\n===== CREATING VISUALIZATIONS =====")
    
    # Create directory if needed
    os.makedirs("visualizations", exist_ok=True)
    
    # Generate sample data
    steps = 100
    train_loss = [5.0 * (0.97**i) + random.uniform(-0.2, 0.2) for i in range(steps)]
    val_loss = [5.5 * (0.96**i) + random.uniform(-0.1, 0.1) for i in range(steps)]
    
    # Create HTML visualization with proper encoding
    with open("visualizations/training_chart.html", "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Training Progress</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; }}
        .chart {{ width: 800px; height: 400px; border: 1px solid #ddd; margin: 20px; }}
        .loss {{ color: #3498db; }}
        .val-loss {{ color: #e74c3c; }}
    </style>
</head>
<body>
    <h1>Training Progress</h1>
    <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="chart" id="lossChart">
        <svg viewBox="0 0 800 400">
            <!-- X Axis -->
            <line x1="50" y1="350" x2="750" y2="350" stroke="#333" />
            
            <!-- Y Axis -->
            <line x1="50" y1="50" x2="50" y2="350" stroke="#333" />
            
            <!-- Training Loss Line -->
            <polyline 
                points="{' '.join([f'{50 + i*7},{350 - val*60}' for i, val in enumerate(train_loss)])}"
                fill="none"
                stroke="#3498db"
                stroke-width="2"
            />
            
            <!-- Validation Loss Line -->
            <polyline 
                points="{' '.join([f'{50 + i*7},{350 - val*60}' for i, val in enumerate(val_loss)])}"
                fill="none"
                stroke="#e74c3c"
                stroke-width="2"
                stroke-dasharray="5,5"
            />
            
            <!-- Legend -->
            <rect x="600" y="50" width="150" height="60" fill="white" stroke="#ccc" />
            <text x="620" y="80" class="loss">Training Loss</text>
            <line x1="610" y1="85" x2="650" y2="85" stroke="#3498db" />
            <text x="620" y="110" class="val-loss">Validation Loss</text>
            <line x1="610" y1="115" x2="650" y2="115" stroke="#e74c3c" stroke-dasharray="5,5" />
        </svg>
    </div>
    
    <h2>Key Metrics</h2>
    <ul>
        <li>Final Training Loss: {train_loss[-1]:.2f}</li>
        <li>Final Validation Loss: {val_loss[-1]:.2f}</li>
        <li>Training Duration: {steps} steps</li>
    </ul>
</body>
</html>""")

    print("✓ Visualization created at visualizations/training_chart.html")
    print("===== VISUALIZATION COMPLETE =====")

if __name__ == "__main__":
    try:
        create_visualizations()
    except Exception as e:
        print(f"Error: {str(e)}")
        with open("visualizations/error_log.txt", "w") as f:
            f.write(f"Error at {time.strftime('%Y-%m-%d %H:%M:%S')}:\n{str(e)}") 