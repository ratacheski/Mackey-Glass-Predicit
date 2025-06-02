#!/bin/bash

# 🌐 GitHub Pages Automatic Setup Script
# Work 2 RNP - Mackey-Glass Prediction
# Author: Rafael Ratacheski de Sousa Raulino

echo "🌐 Setting up GitHub Pages for Work 2 RNP..."
echo "=" * 60
echo "📝 Developed by: Rafael Ratacheski de Sousa Raulino"
echo "🎓 MSc Student in Electrical and Computer Engineering - UFG"
echo "📚 Course: Deep Neural Networks - 2025/1"
echo "=" * 60

# Check if we are in the correct directory
if [ ! -d "mackey_glass_prediction" ]; then
    echo "❌ Error: Run this script from the project root (where the mackey_glass_prediction folder is located)"
    exit 1
fi

# Create GitHub Pages structure
echo "📁 Creating GitHub Pages structure..."
mkdir -p docs/images
mkdir -p docs/css
mkdir -p docs/js

# Find the most recent results folder
RESULTS_DIR=$(find mackey_glass_prediction/experiments/results -name "final_report_*" -type d | sort | tail -1)

if [ -z "$RESULTS_DIR" ]; then
    echo "❌ No results folder found!"
    echo "💡 First run:"
    echo "   cd mackey_glass_prediction/experiments"
    echo "   python run_experiment.py"
    echo "   cd ../.."
    echo "   python mackey_glass_prediction/generate_interactive_report.py"
    exit 1
fi

echo "📁 Copying results from: $RESULTS_DIR"

# Copy results files
echo "📊 Copying images and reports..."
cp -r "$RESULTS_DIR"/* docs/images/ 2>/dev/null || true

# Copy main HTML report
if [ -f "$RESULTS_DIR/report.html" ]; then
    cp "$RESULTS_DIR/report.html" docs/report.html
    echo "✅ HTML report copied"
else
    # Try to generate report if it doesn't exist
    echo "📄 HTML report not found. Generating..."
    cd mackey_glass_prediction
    python generate_interactive_report.py
    cd ..
    
    # Try again
    if [ -f "$RESULTS_DIR/report.html" ]; then
        cp "$RESULTS_DIR/report.html" docs/report.html
        echo "✅ HTML report generated and copied"
    else
        echo "⚠️ Could not find/generate HTML report"
    fi
fi

# Adjust paths in HTML - Improved Version
if [ -f "docs/report.html" ]; then
    echo "🔧 Using optimized script for path correction..."
    
    # Check if the correction script exists
    if [ -f "fix_image_paths.sh" ]; then
        echo "✅ Running fix_image_paths.sh..."
        chmod +x fix_image_paths.sh
        ./fix_image_paths.sh
    else
        echo "⚠️ Script fix_image_paths.sh not found!"
        echo "🔧 Using basic method (only fixes src, not onclick)..."
        
        # Backup the original file
        cp docs/report.html docs/report.html.backup
        
        # Debug: show some path examples before conversion
        echo "🔍 Example paths found:"
        grep -o 'src="[^"]*\.\(png\|jpg\|jpeg\|gif\|svg\)"' docs/report.html | head -3
        
        # Main Strategy: Replace direct files (without folder)
        # Example: src="arquivo.png" -> src="images/arquivo.png"
        echo "🔄 Applying main correction for direct files..."
        sed -i 's@src="\([^"/]*\.\(png\|jpg\|jpeg\|gif\|svg\)\)"@src="images/\1"@g' docs/report.html
        
        # Strategy 2: For files that may already have a folder
        # Example: src="pasta/arquivo.png" -> src="images/pasta/arquivo.png"
        sed -i 's@src="\([^"]*[^/]\)/\([^"]*\.\(png\|jpg\|jpeg\|gif\|svg\)\)"@src="images/\1/\2"@g' docs/report.html
        
        # Strategy 3: For absolute paths
        # Example: src="/caminho/arquivo.png" -> src="images/caminho/arquivo.png"
        sed -i 's@src="/\([^"]*\.\(png\|jpg\|jpeg\|gif\|svg\)\)"@src="images/\1"@g' docs/report.html
        
        # Apply the same for href (links to files)
        echo "🔗 Fixing href links..."
        sed -i 's@href="\([^"/]*\.\(png\|jpg\|jpeg\|gif\|svg\|csv\|txt\)\)"@href="images/\1"@g' docs/report.html
        sed -i 's@href="\([^"]*[^/]\)/\([^"]*\.\(png\|jpg\|jpeg\|gif\|svg\|csv\|txt\)\)"@href="images/\1/\2"@g' docs/report.html
        sed -i 's@href="/\([^"]*\.\(png\|jpg\|jpeg\|gif\|svg\|csv\|txt\)\)"@href="images/\1"@g' docs/report.html
        
        # Fix duplications that may have occurred
        echo "🧹 Removing duplications..."
        sed -i 's@images/images/@images/@g' docs/report.html
        sed -i 's@src="images/images/@src="images/@g' docs/report.html
        sed -i 's@href="images/images/@href="images/@g' docs/report.html
        
        # Debug: show some examples after conversion
        echo "🔄 Example paths after conversion:"
        grep -o 'src="[^"]*\.\(png\|jpg\|jpeg\|gif\|svg\)"' docs/report.html | head -3
        
        # Check if there were changes by comparing with backup
        if ! diff -q docs/report.html.backup docs/report.html > /dev/null; then
            echo "✅ Paths adjusted successfully (basic method)"
            
            # Show statistics
            total_imgs=$(grep -o 'src="[^"]*\.\(png\|jpg\|jpeg\|gif\|svg\)"' docs/report.html | wc -l)
            correct_imgs=$(grep -o 'src="images/[^"]*\.\(png\|jpg\|jpeg\|gif\|svg\)"' docs/report.html | wc -l)
            echo "📊 Statistics: $correct_imgs/$total_imgs images with correct paths"
            echo "⚠️ Note: This method does not fix onclick events!"
            echo "💡 For complete correction, run: ./fix_image_paths.sh"
            
            rm docs/report.html.backup
        else
            echo "⚠️ No changes were made to the paths"
            echo "💡 Check if paths are already correct or run:"
            echo "   ./debug_image_paths.sh"
            rm docs/report.html.backup
        fi
    fi
fi

# Create main page (index.html)
echo "🏠 Creating main page..."
cat > docs/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Work 2 RNP - Mackey-Glass Prediction</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        .container {
            background: rgba(255, 255, 255, 0.95);
            color: #333;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: white;
            padding: 25px;
            margin: 20px 0;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            border-left: 5px solid #3498db;
        }
        .btn {
            display: inline-block;
            background: #3498db;
            color: white;
            padding: 15px 30px;
            text-decoration: none;
            border-radius: 5px;
            margin: 10px 5px;
            transition: all 0.3s ease;
            font-weight: bold;
        }
        .btn:hover {
            background: #2980b9;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }
        .btn-success {
            background: #27ae60;
            font-size: 1.1em;
        }
        .btn-success:hover {
            background: #229954;
        }
        .author-info {
            background: #ecf0f1;
            padding: 20px;
            border-radius: 10px;
            margin-top: 30px;
            text-align: center;
        }
        .author-info h3 {
            color: #2c3e50;
            margin-bottom: 10px;
        }
        .author-info p {
            color: #7f8c8d;
            margin: 5px 0;
        }
        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        .highlight {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 3px 8px;
            border-radius: 5px;
            font-weight: bold;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
            color: #7f8c8d;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Work 2 - Deep Neural Networks</h1>
        <h2 style="text-align: center; color: #7f8c8d; margin-bottom: 30px;">Mackey-Glass Time Series Prediction</h2>
        
        <div class="card">
            <h3>📊 About the Project</h3>
            <p>This project implements and compares <span class="highlight">three types of neural networks</span> (MLP, LSTM, GRU) for Mackey-Glass time series prediction, with multiple configurations and variations for comprehensive analysis.</p>
            
            <div class="results-grid">
                <div class="metric-card">
                    <h4>🏆 Best Model</h4>
                    <div class="metric-value">LSTM Bidirectional</div>
                    <p>R² = 0.990789</p>
                </div>
                <div class="metric-card">
                    <h4>🔬 Models Evaluated</h4>
                    <div class="metric-value">7</div>
                    <p>Optimized Configurations</p>
                </div>
                <div class="metric-card">
                    <h4>📈 Dataset</h4>
                    <div class="metric-value">998</div>
                    <p>Validation Points</p>
                </div>
            </div>
        </div>

        <div class="card">
            <h3>🌐 Interactive Report</h3>
            <p>Access the complete report with <strong>interactive visualizations</strong>, detailed metrics (EQMN1, EQMN2, R², RMSE, MAE, MAPE) and statistical analyses of the conducted experiments.</p>
            <div style="text-align: center; margin-top: 20px;">
                <a href="report.html" class="btn btn-success">📊 View Complete Interactive Report</a>
            </div>
        </div>

        <div class="card">
            <h3>📚 Documentation</h3>
            <p>Explore the complete project documentation:</p>
            <a href="https://github.com/ratacheski/Mackey-Glass-Predicit/blob/master/README.md" class="btn">📖 README</a>
            <a href="https://github.com/ratacheski/Mackey-Glass-Predicit/blob/master/HOW_TO_USE.md" class="btn">🚀 How to Use</a>
            <a href="https://github.com/ratacheski/Mackey-Glass-Predicit/blob/master/FINAL_RESULTS.md" class="btn">📈 Results</a>
            <a href="https://github.com/ratacheski/Mackey-Glass-Predicit/blob/master/EXECUTIVE_SUMMARY.md" class="btn">📊 Summary</a>
        </div>

        <div class="card">
            <h3>📁 Source Code</h3>
            <p>Access the complete and reproducible code on GitHub:</p>
            <a href="https://github.com/ratacheski/Mackey-Glass-Predicit" class="btn">💻 GitHub Repository</a>
        </div>

        <div class="author-info">
            <h3>👨‍🎓 Author</h3>
            <p><strong>Rafael Ratacheski de Sousa Raulino</strong></p>
            <p>MSc Student in Electrical and Computer Engineering - UFG</p>
            <p>Course: Deep Neural Networks - 2025/1</p>
            <p>Date: June 2025</p>
        </div>

        <div class="footer">
            <p>🌐 Hosted on GitHub Pages | 🔬 Experiments conducted with PyTorch</p>
        </div>
    </div>
</body>
</html>
EOF

# Request user information
echo ""
echo "🔧 Custom configuration:"

echo "✅ Main page created"

# Create custom CSS file
echo "🎨 Creating custom styles..."
cat > docs/css/styles.css << 'EOF'
/* Additional styles for GitHub Pages */
.github-badge {
    position: fixed;
    top: 0;
    right: 0;
    background: #333;
    color: white;
    padding: 10px 20px;
    text-decoration: none;
    z-index: 1000;
    border-bottom-left-radius: 5px;
}

.github-badge:hover {
    background: #555;
    color: white;
}

/* Responsive design */
@media (max-width: 768px) {
    .container {
        padding: 20px;
        margin: 10px;
    }
    
    .results-grid {
        grid-template-columns: 1fr;
    }
    
    .btn {
        display: block;
        margin: 10px 0;
        text-align: center;
    }
}
EOF

# Check created files
echo ""
echo "📊 Checking created files:"
echo "📁 docs/"
echo "   ├── index.html ✅"
echo "   ├── report.html $([ -f docs/report.html ] && echo "✅" || echo "❌")"
echo "   ├── css/styles.css ✅"
echo "   └── images/ $([ -d docs/images ] && echo "✅ ($(ls docs/images/ 2>/dev/null | wc -l) files)" || echo "❌")"

# Count image files
IMG_COUNT=$(find docs/images -type f 2>/dev/null | wc -l)
echo "🖼️ Total files copied: $IMG_COUNT"

echo ""
echo "✅ GitHub Pages setup completed!"
echo ""
echo "📋 NEXT STEPS:"
echo "   1. git add docs/"
echo "   2. git commit -m 'Add GitHub Pages configuration with interactive report'"
echo "   3. git push origin master"
echo "   4. Go to Settings → Pages on GitHub"
echo "   5. Configure Source: 'Deploy from a branch'"
echo "   6. Branch: 'master', Folder: '/docs'"
echo "   7. Wait for deployment (2-3 minutes)"
echo ""
echo "🌐 Final URL will be:"
echo "   https://ratacheski.github.io/Mackey-Glass-Predicit/"
echo ""
echo "📊 Interactive report at:"
echo "   https://ratacheski.github.io/Mackey-Glass-Predicit/report.html"
echo ""
echo "💡 TIP: Add this badge to your README.md:"
echo "[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live%20Demo-brightgreen?style=for-the-badge&logo=github)](https://ratacheski.github.io/Mackey-Glass-Predicit/)" 