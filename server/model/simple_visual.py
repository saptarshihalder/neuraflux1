import os
import sys
import random
import time

def simple_ascii_plot(data, title, width=40, height=10):
    """Create a simple ASCII plot with no dependencies"""
    max_val = max(data)
    min_val = min(data)
    range_val = max_val - min_val if max_val != min_val else 1
    
    print(f"\n{title}")
    print("=" * width)
    
    for i in range(height, 0, -1):
        row = ""
        threshold = min_val + (i / height) * range_val
        for val in data:
            if val >= threshold:
                row += "#"
            else:
                row += " "
        print(f"{row} {threshold:.2f}")
    
    print("=" * width)
    print("".join([str(i % 10) for i in range(min(width, len(data)))]))
    print(f"Min: {min_val:.2f}, Max: {max_val:.2f}")

def create_simple_visualizations():
    """Create simple visualizations without matplotlib"""
    print("\n===== CREATING SIMPLE VISUALIZATIONS =====")
    
    # Create output directory
    os.makedirs("visualizations", exist_ok=True)
    
    # Generate sample data
    steps = 40
    
    # Training loss (decaying exponential with noise)
    train_loss = []
    loss = 5.0
    for i in range(steps):
        loss = loss * 0.92 + random.uniform(-0.2, 0.2)
        train_loss.append(loss)
    
    # Validation loss (slightly higher, less noise)
    val_loss = []
    loss = 5.5
    for i in range(steps):
        loss = loss * 0.93 + random.uniform(-0.1, 0.1)
        val_loss.append(loss)
    
    # Learning rate (step decay)
    lr = []
    current_lr = 0.001
    for i in range(steps):
        if i > 0 and i % 10 == 0:
            current_lr *= 0.5
        lr.append(current_lr)
    
    # Rewards (gradually increasing)
    rewards = []
    reward = -0.5
    for i in range(steps):
        reward = reward + 0.05 + random.uniform(-0.05, 0.05)
        rewards.append(min(reward, 1.0))  # Cap at 1.0
    
    # Create ASCII plots
    simple_ascii_plot(train_loss, "Training Loss")
    simple_ascii_plot(val_loss, "Validation Loss")
    simple_ascii_plot(lr, "Learning Rate")
    simple_ascii_plot(rewards, "Rewards")
    
    # Save data to CSV
    with open("visualizations/training_metrics.csv", "w") as f:
        f.write("step,train_loss,val_loss,learning_rate,reward\n")
        for i in range(steps):
            f.write(f"{i},{train_loss[i]:.4f},{val_loss[i]:.4f},{lr[i]:.6f},{rewards[i]:.4f}\n")
    
    # Create a simple HTML visualization
    with open("visualizations/simple_chart.html", "w") as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Simple Training Visualization</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .chart { width: 80%; height: 300px; border: 1px solid #ddd; position: relative; margin: 20px 0; }
        .bar { position: absolute; bottom: 0; background: #3498db; width: 8px; }
        .title { font-size: 18px; font-weight: bold; margin-bottom: 10px; }
    </style>
</head>
<body>
    <h1>NeuraFlux Training Visualization</h1>
    <p>Generated at: """ + time.strftime('%Y-%m-%d %H:%M:%S') + """</p>
    
    <div class="title">Training Loss</div>
    <div class="chart" id="train-loss">
""")
        
        for i, loss in enumerate(train_loss):
            height = int((loss / max(train_loss)) * 250)
            left = (i / len(train_loss)) * 80
            f.write(f'        <div class="bar" style="height: {height}px; left: {left}%;"></div>\n')
        
        f.write("""
    </div>
    
    <div class="title">Rewards</div>
    <div class="chart" id="rewards">
""")
        
        for i, reward in enumerate(rewards):
            height = int((reward - min(rewards)) / (max(rewards) - min(rewards)) * 250)
            left = (i / len(rewards)) * 80
            f.write(f'        <div class="bar" style="height: {height}px; left: {left}%; background: #2ecc71;"></div>\n')
        
        f.write("""
    </div>
    
    <p>These charts are generated without any external libraries.</p>
    <p>To generate proper visualizations, install matplotlib and other required libraries.</p>
</body>
</html>
""")
    
    print(f"\n✓ Data saved to visualizations/training_metrics.csv")
    print(f"✓ HTML visualization created at visualizations/simple_chart.html")
    print(f"✓ Open the HTML file in a browser to see interactive charts")
    print("===== VISUALIZATION COMPLETE =====\n")

if __name__ == "__main__":
    try:
        print(f"Running script with Python: {sys.executable}")
        create_simple_visualizations()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        # Write error to file for debugging
        with open("error_log.txt", "w") as f:
            f.write(f"Error at {time.strftime('%Y-%m-%d %H:%M:%S')}:\n{str(e)}") 