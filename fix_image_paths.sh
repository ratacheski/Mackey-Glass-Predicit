#!/bin/bash

# 🛠️ Image Path Correction Script
# Work 2 RNP - Mackey-Glass Prediction

echo "🛠️ Fixing Image Paths for GitHub Pages"
echo "=" * 50

# Check if the HTML file exists
if [ ! -f "docs/report.html" ]; then
    echo "❌ File docs/report.html not found!"
    echo "First run: ./setup_github_pages.sh"
    exit 1
fi

# Make backup
echo "💾 Creating backup of HTML file..."
cp docs/report.html docs/report.html.backup.$(date +%Y%m%d_%H%M%S)

echo "🔧 Applying path corrections..."

# Strategy 1 - Main: Direct files (without folder)
echo "📌 Strategy 1: Fixing direct files"
echo "   Example: src=\"file.png\" -> src=\"images/file.png\""
sed -i 's@src="\([^"/]*\.\(png\|jpg\|jpeg\|gif\|svg\)\)"@src="images/\1"@g' docs/report.html

# Strategy 2: Paths with one folder
echo "📌 Strategy 2: Fixing paths with folder"
echo "   Example: src=\"folder/file.png\" -> src=\"images/folder/file.png\""
sed -i 's@src="\([^"/][^"]*\)/\([^"]*\.\(png\|jpg\|jpeg\|gif\|svg\)\)"@src="images/\1/\2"@g' docs/report.html

# Strategy 3: Absolute paths
echo "📌 Strategy 3: Fixing absolute paths"
echo "   Example: src=\"/file.png\" -> src=\"images/file.png\""
sed -i 's@src="/\([^"]*\.\(png\|jpg\|jpeg\|gif\|svg\)\)"@src="images/\1"@g' docs/report.html

# Strategy 4: href links
echo "📌 Strategy 4: Fixing href links"
sed -i 's@href="\([^"/]*\.\(png\|jpg\|jpeg\|gif\|svg\|csv\|txt\)\)"@href="images/\1"@g' docs/report.html
sed -i 's@href="\([^"/][^"]*\)/\([^"]*\.\(png\|jpg\|jpeg\|gif\|svg\|csv\|txt\)\)"@href="images/\1/\2"@g' docs/report.html
sed -i 's@href="/\([^"]*\.\(png\|jpg\|jpeg\|gif\|svg\|csv\|txt\)\)"@href="images/\1"@g' docs/report.html

# *** NEW STRATEGY *** - Fix paths in onclick (IDENTIFIED PROBLEM)
echo "📌 Strategy 4B: Fixing paths in onclick"
echo "   Example: onclick=\"openModal('file.png', ...)\" -> onclick=\"openModal('images/file.png', ...)\""
sed -i "s@onclick=\"openModal('\([^'/]*\.\(png\|jpg\|jpeg\|gif\|svg\)\)',@onclick=\"openModal('images/\1',@g" docs/report.html
sed -i "s@onclick=\"openModal('\([^'/][^']*\)/\([^']*\.\(png\|jpg\|jpeg\|gif\|svg\)\)',@onclick=\"openModal('images/\1/\2',@g" docs/report.html
sed -i "s@onclick=\"openModal('/\([^']*\.\(png\|jpg\|jpeg\|gif\|svg\)\)',@onclick=\"openModal('images/\1',@g" docs/report.html

# *** NEW STRATEGY *** - Fix other types of JavaScript events
echo "📌 Strategy 4C: Fixing other JavaScript events"
sed -i "s@'\([^'/]*\.\(png\|jpg\|jpeg\|gif\|svg\)\)'@'images/\1'@g" docs/report.html

# Strategy 5: Remove duplications
echo "📌 Strategy 5: Removing duplications"
sed -i 's@images/images/@images/@g' docs/report.html
sed -i 's@src="images/images/@src="images/@g' docs/report.html
sed -i 's@href="images/images/@href="images/@g' docs/report.html
sed -i "s@'images/images/@'images/@g" docs/report.html

# Strategy 6: Specific corrections for model folders
echo "📌 Strategy 6: Specific corrections"
sed -i 's@src="\(mlp_[^"]*\)\.png"@src="images/\1.png"@g' docs/report.html
sed -i 's@src="\(lstm_[^"]*\)\.png"@src="images/\1.png"@g' docs/report.html
sed -i 's@src="\(gru_[^"]*\)\.png"@src="images/\1.png"@g' docs/report.html
sed -i 's@src="\(rnn_[^"]*\)\.png"@src="images/\1.png"@g' docs/report.html

# Strategy 7: Final cleanup
echo "📌 Strategy 7: Final cleanup"
sed -i 's@//images/@/images/@g' docs/report.html
sed -i 's@///images/@/images/@g' docs/report.html

echo ""
echo "🔍 Checking results..."

# Count corrected paths in src
total_images=$(grep -o 'src="[^"]*\.\(png\|jpg\|jpeg\|gif\|svg\)' docs/report.html | wc -l)
correct_src_paths=$(grep -o 'src="images/' docs/report.html | wc -l)

# Count corrected paths in onclick
total_onclick=$(grep -o "onclick=\"openModal('[^']*\.\(png\|jpg\|jpeg\|gif\|svg\)'" docs/report.html | wc -l)
correct_onclick_paths=$(grep -o "onclick=\"openModal('images/" docs/report.html | wc -l)

echo "📊 Correction results:"
echo "   📸 SRC - Total images: $total_images"
echo "   📸 SRC - Correct paths: $correct_src_paths"
echo "   🖱️ ONCLICK - Total events: $total_onclick"
echo "   🖱️ ONCLICK - Correct paths: $correct_onclick_paths"

# Check remaining problems
incorrect_src=$((total_images - correct_src_paths))
incorrect_onclick=$((total_onclick - correct_onclick_paths))

if [ $incorrect_src -eq 0 ] && [ $incorrect_onclick -eq 0 ]; then
    echo "🎉 ✅ All paths were corrected successfully!"
else
    echo "⚠️ Problems still exist:"
    [ $incorrect_src -gt 0 ] && echo "   - $incorrect_src incorrect SRC paths"
    [ $incorrect_onclick -gt 0 ] && echo "   - $incorrect_onclick incorrect ONCLICK paths"
    
    echo ""
    echo "🔍 Examples of remaining problems:"
    
    if [ $incorrect_src -gt 0 ]; then
        echo "   📸 Problematic SRC:"
        grep -n 'src="[^"]*\.\(png\|jpg\|jpeg\|gif\|svg\)' docs/report.html | grep -v 'src="images/' | head -2
    fi
    
    if [ $incorrect_onclick -gt 0 ]; then
        echo "   🖱️ Problematic ONCLICK:"
        grep -n "onclick=\"openModal('[^']*\.\(png\|jpg\|jpeg\|gif\|svg\)'" docs/report.html | grep -v "onclick=\"openModal('images/" | head -2
    fi
    
    echo ""
    echo "💡 Tips for manual correction:"
    echo "1. Check if all files are in docs/images/"
    echo "2. Open docs/report.html and look for paths without 'images/'"
    echo "3. Run: ./debug_image_paths.sh for more details"
fi

echo ""
echo "📋 Next steps:"
echo "1. Run: ./debug_image_paths.sh to check the result"
echo "2. Test locally by opening docs/index.html in browser"
echo "3. If everything is ok, commit:"
echo "   git add docs/"
echo "   git commit -m 'Fix image paths for GitHub Pages'"
echo "   git push"

echo ""
echo "💾 Backup created at: docs/report.html.backup.$(date +%Y%m%d_%H%M%S)" 