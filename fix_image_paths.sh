#!/bin/bash

# 🛠️ Script de Correção de Caminhos de Imagens
# Trabalho 2 RNP - Predição Mackey-Glass

echo "🛠️ Corrigindo Caminhos de Imagens para GitHub Pages"
echo "=" * 50

# Verificar se existe o arquivo HTML
if [ ! -f "docs/relatorio.html" ]; then
    echo "❌ Arquivo docs/relatorio.html não encontrado!"
    echo "Execute primeiro: ./setup_github_pages.sh"
    exit 1
fi

# Fazer backup
echo "💾 Criando backup do arquivo HTML..."
cp docs/relatorio.html docs/relatorio.html.backup.$(date +%Y%m%d_%H%M%S)

echo "🔧 Aplicando correções de caminhos..."

# Estratégia 1 - Principal: Arquivos diretos (sem pasta)
echo "📌 Estratégia 1: Corrigindo arquivos diretos"
echo "   Exemplo: src=\"arquivo.png\" -> src=\"images/arquivo.png\""
sed -i 's@src="\([^"/]*\.\(png\|jpg\|jpeg\|gif\|svg\)\)"@src="images/\1"@g' docs/relatorio.html

# Estratégia 2: Caminhos com uma pasta
echo "📌 Estratégia 2: Corrigindo caminhos com pasta"
echo "   Exemplo: src=\"pasta/arquivo.png\" -> src=\"images/pasta/arquivo.png\""
sed -i 's@src="\([^"/][^"]*\)/\([^"]*\.\(png\|jpg\|jpeg\|gif\|svg\)\)"@src="images/\1/\2"@g' docs/relatorio.html

# Estratégia 3: Caminhos absolutos
echo "📌 Estratégia 3: Corrigindo caminhos absolutos"
echo "   Exemplo: src=\"/arquivo.png\" -> src=\"images/arquivo.png\""
sed -i 's@src="/\([^"]*\.\(png\|jpg\|jpeg\|gif\|svg\)\)"@src="images/\1"@g' docs/relatorio.html

# Estratégia 4: Links href
echo "📌 Estratégia 4: Corrigindo links href"
sed -i 's@href="\([^"/]*\.\(png\|jpg\|jpeg\|gif\|svg\|csv\|txt\)\)"@href="images/\1"@g' docs/relatorio.html
sed -i 's@href="\([^"/][^"]*\)/\([^"]*\.\(png\|jpg\|jpeg\|gif\|svg\|csv\|txt\)\)"@href="images/\1/\2"@g' docs/relatorio.html
sed -i 's@href="/\([^"]*\.\(png\|jpg\|jpeg\|gif\|svg\|csv\|txt\)\)"@href="images/\1"@g' docs/relatorio.html

# *** NOVA ESTRATÉGIA *** - Corrigir caminhos em onclick (PROBLEMA IDENTIFICADO)
echo "📌 Estratégia 4B: Corrigindo caminhos em onclick"
echo "   Exemplo: onclick=\"openModal('arquivo.png', ...)\" -> onclick=\"openModal('images/arquivo.png', ...)\""
sed -i "s@onclick=\"openModal('\([^'/]*\.\(png\|jpg\|jpeg\|gif\|svg\)\)',@onclick=\"openModal('images/\1',@g" docs/relatorio.html
sed -i "s@onclick=\"openModal('\([^'/][^']*\)/\([^']*\.\(png\|jpg\|jpeg\|gif\|svg\)\)',@onclick=\"openModal('images/\1/\2',@g" docs/relatorio.html
sed -i "s@onclick=\"openModal('/\([^']*\.\(png\|jpg\|jpeg\|gif\|svg\)\)',@onclick=\"openModal('images/\1',@g" docs/relatorio.html

# *** NOVA ESTRATÉGIA *** - Corrigir outros tipos de eventos de JavaScript
echo "📌 Estratégia 4C: Corrigindo outros eventos JavaScript"
sed -i "s@'\([^'/]*\.\(png\|jpg\|jpeg\|gif\|svg\)\)'@'images/\1'@g" docs/relatorio.html

# Estratégia 5: Remover duplicações
echo "📌 Estratégia 5: Removendo duplicações"
sed -i 's@images/images/@images/@g' docs/relatorio.html
sed -i 's@src="images/images/@src="images/@g' docs/relatorio.html
sed -i 's@href="images/images/@href="images/@g' docs/relatorio.html
sed -i "s@'images/images/@'images/@g" docs/relatorio.html

# Estratégia 6: Correções específicas para pastas de modelos
echo "📌 Estratégia 6: Correções específicas"
sed -i 's@src="\(mlp_[^"]*\)\.png"@src="images/\1.png"@g' docs/relatorio.html
sed -i 's@src="\(lstm_[^"]*\)\.png"@src="images/\1.png"@g' docs/relatorio.html
sed -i 's@src="\(gru_[^"]*\)\.png"@src="images/\1.png"@g' docs/relatorio.html
sed -i 's@src="\(rnn_[^"]*\)\.png"@src="images/\1.png"@g' docs/relatorio.html

# Estratégia 7: Limpeza final
echo "📌 Estratégia 7: Limpeza final"
sed -i 's@//images/@/images/@g' docs/relatorio.html
sed -i 's@///images/@/images/@g' docs/relatorio.html

echo ""
echo "🔍 Verificando resultados..."

# Contar caminhos corrigidos em src
total_images=$(grep -o 'src="[^"]*\.\(png\|jpg\|jpeg\|gif\|svg\)' docs/relatorio.html | wc -l)
correct_src_paths=$(grep -o 'src="images/' docs/relatorio.html | wc -l)

# Contar caminhos corrigidos em onclick
total_onclick=$(grep -o "onclick=\"openModal('[^']*\.\(png\|jpg\|jpeg\|gif\|svg\)'" docs/relatorio.html | wc -l)
correct_onclick_paths=$(grep -o "onclick=\"openModal('images/" docs/relatorio.html | wc -l)

echo "📊 Resultados da correção:"
echo "   📸 SRC - Total de imagens: $total_images"
echo "   📸 SRC - Caminhos corretos: $correct_src_paths"
echo "   🖱️ ONCLICK - Total de eventos: $total_onclick"
echo "   🖱️ ONCLICK - Caminhos corretos: $correct_onclick_paths"

# Verificar problemas restantes
incorrect_src=$((total_images - correct_src_paths))
incorrect_onclick=$((total_onclick - correct_onclick_paths))

if [ $incorrect_src -eq 0 ] && [ $incorrect_onclick -eq 0 ]; then
    echo "🎉 ✅ Todos os caminhos foram corrigidos com sucesso!"
else
    echo "⚠️ Ainda existem problemas:"
    [ $incorrect_src -gt 0 ] && echo "   - $incorrect_src caminhos SRC incorretos"
    [ $incorrect_onclick -gt 0 ] && echo "   - $incorrect_onclick caminhos ONCLICK incorretos"
    
    echo ""
    echo "🔍 Exemplos de problemas restantes:"
    
    if [ $incorrect_src -gt 0 ]; then
        echo "   📸 SRC problemáticos:"
        grep -n 'src="[^"]*\.\(png\|jpg\|jpeg\|gif\|svg\)' docs/relatorio.html | grep -v 'src="images/' | head -2
    fi
    
    if [ $incorrect_onclick -gt 0 ]; then
        echo "   🖱️ ONCLICK problemáticos:"
        grep -n "onclick=\"openModal('[^']*\.\(png\|jpg\|jpeg\|gif\|svg\)'" docs/relatorio.html | grep -v "onclick=\"openModal('images/" | head -2
    fi
    
    echo ""
    echo "💡 Dicas para correção manual:"
    echo "1. Verifique se todos os arquivos estão em docs/images/"
    echo "2. Abra docs/relatorio.html e procure por caminhos sem 'images/'"
    echo "3. Execute: ./debug_image_paths.sh para mais detalhes"
fi

echo ""
echo "📋 Próximos passos:"
echo "1. Execute: ./debug_image_paths.sh para verificar o resultado"
echo "2. Teste localmente abrindo docs/index.html no browser"
echo "3. Se estiver tudo ok, faça commit:"
echo "   git add docs/"
echo "   git commit -m 'Fix image paths for GitHub Pages'"
echo "   git push"

echo ""
echo "💾 Backup criado em: docs/relatorio.html.backup.$(date +%Y%m%d_%H%M%S)" 