#!/bin/bash

# 🌐 Script de Configuração Automática do GitHub Pages
# Trabalho 2 RNP - Predição Mackey-Glass
# Autor: Rafael Ratacheski de Sousa Raulino

echo "🌐 Configurando GitHub Pages para Trabalho 2 RNP..."
echo "=" * 60
echo "📝 Desenvolvido por: Rafael Ratacheski de Sousa Raulino"
echo "🎓 Mestrando em Engenharia Elétrica e de Computação - UFG"
echo "📚 Disciplina: Redes Neurais Profundas - 2025/1"
echo "=" * 60

# Verificar se estamos no diretório correto
if [ ! -d "mackey_glass_prediction" ]; then
    echo "❌ Erro: Execute este script a partir da raiz do projeto (onde está a pasta mackey_glass_prediction)"
    exit 1
fi

# Criar estrutura do GitHub Pages
echo "📁 Criando estrutura do GitHub Pages..."
mkdir -p docs/images
mkdir -p docs/css
mkdir -p docs/js

# Encontrar pasta de resultados mais recente
RESULTS_DIR=$(find mackey_glass_prediction/experiments/results -name "final_report_*" -type d | sort | tail -1)

if [ -z "$RESULTS_DIR" ]; then
    echo "❌ Nenhuma pasta de resultados encontrada!"
    echo "💡 Execute primeiro:"
    echo "   cd mackey_glass_prediction/experiments"
    echo "   python run_experiment.py"
    echo "   cd ../.."
    echo "   python mackey_glass_prediction/generate_interactive_report.py"
    exit 1
fi

echo "📁 Copiando resultados de: $RESULTS_DIR"

# Copiar arquivos de resultados
echo "📊 Copiando imagens e relatórios..."
cp -r "$RESULTS_DIR"/* docs/images/ 2>/dev/null || true

# Copiar relatório HTML principal
if [ -f "$RESULTS_DIR/relatorio.html" ]; then
    cp "$RESULTS_DIR/relatorio.html" docs/relatorio.html
    echo "✅ Relatório HTML copiado"
else
    # Tentar gerar relatório se não existir
    echo "📄 Relatório HTML não encontrado. Gerando..."
    cd mackey_glass_prediction
    python generate_interactive_report.py
    cd ..
    
    # Tentar novamente
    if [ -f "$RESULTS_DIR/relatorio.html" ]; then
        cp "$RESULTS_DIR/relatorio.html" docs/relatorio.html
        echo "✅ Relatório HTML gerado e copiado"
    else
        echo "⚠️ Não foi possível encontrar/gerar o relatório HTML"
    fi
fi

# Ajustar caminhos no HTML - Versão Melhorada
if [ -f "docs/relatorio.html" ]; then
    echo "🔧 Usando script otimizado para correção de caminhos..."
    
    # Verificar se o script de correção existe
    if [ -f "fix_image_paths.sh" ]; then
        echo "✅ Executando fix_image_paths.sh..."
        chmod +x fix_image_paths.sh
        ./fix_image_paths.sh
    else
        echo "⚠️ Script fix_image_paths.sh não encontrado!"
        echo "🔧 Usando método básico (só corrige src, não onclick)..."
        
        # Backup do arquivo original
        cp docs/relatorio.html docs/relatorio.html.backup
        
        # Debug: mostrar alguns exemplos de caminhos antes da conversão
        echo "🔍 Exemplo de caminhos encontrados:"
        grep -o 'src="[^"]*\.\(png\|jpg\|jpeg\|gif\|svg\)"' docs/relatorio.html | head -3
        
        # Estratégia Principal: Substituir arquivos diretos (sem pasta)
        # Exemplo: src="arquivo.png" -> src="images/arquivo.png"
        echo "🔄 Aplicando correção principal para arquivos diretos..."
        sed -i 's@src="\([^"/]*\.\(png\|jpg\|jpeg\|gif\|svg\)\)"@src="images/\1"@g' docs/relatorio.html
        
        # Estratégia 2: Para arquivos que já podem ter uma pasta
        # Exemplo: src="pasta/arquivo.png" -> src="images/pasta/arquivo.png"
        sed -i 's@src="\([^"]*[^/]\)/\([^"]*\.\(png\|jpg\|jpeg\|gif\|svg\)\)"@src="images/\1/\2"@g' docs/relatorio.html
        
        # Estratégia 3: Para caminhos absolutos
        # Exemplo: src="/caminho/arquivo.png" -> src="images/caminho/arquivo.png"
        sed -i 's@src="/\([^"]*\.\(png\|jpg\|jpeg\|gif\|svg\)\)"@src="images/\1"@g' docs/relatorio.html
        
        # Aplicar o mesmo para href (links para arquivos)
        echo "🔗 Corrigindo links href..."
        sed -i 's@href="\([^"/]*\.\(png\|jpg\|jpeg\|gif\|svg\|csv\|txt\)\)"@href="images/\1"@g' docs/relatorio.html
        sed -i 's@href="\([^"]*[^/]\)/\([^"]*\.\(png\|jpg\|jpeg\|gif\|svg\|csv\|txt\)\)"@href="images/\1/\2"@g' docs/relatorio.html
        sed -i 's@href="/\([^"]*\.\(png\|jpg\|jpeg\|gif\|svg\|csv\|txt\)\)"@href="images/\1"@g' docs/relatorio.html
        
        # Corrigir duplicações que podem ter ocorrido
        echo "🧹 Removendo duplicações..."
        sed -i 's@images/images/@images/@g' docs/relatorio.html
        sed -i 's@src="images/images/@src="images/@g' docs/relatorio.html
        sed -i 's@href="images/images/@href="images/@g' docs/relatorio.html
        
        # Debug: mostrar alguns exemplos após a conversão
        echo "🔄 Exemplo de caminhos após conversão:"
        grep -o 'src="[^"]*\.\(png\|jpg\|jpeg\|gif\|svg\)"' docs/relatorio.html | head -3
        
        # Verificar se houve mudanças comparando com backup
        if ! diff -q docs/relatorio.html.backup docs/relatorio.html > /dev/null; then
            echo "✅ Caminhos ajustados com sucesso (método básico)"
            
            # Mostrar estatísticas
            total_imgs=$(grep -o 'src="[^"]*\.\(png\|jpg\|jpeg\|gif\|svg\)"' docs/relatorio.html | wc -l)
            correct_imgs=$(grep -o 'src="images/[^"]*\.\(png\|jpg\|jpeg\|gif\|svg\)"' docs/relatorio.html | wc -l)
            echo "📊 Estatísticas: $correct_imgs/$total_imgs imagens com caminhos corretos"
            echo "⚠️ Nota: Este método não corrige eventos onclick!"
            echo "💡 Para correção completa, execute: ./fix_image_paths.sh"
            
            rm docs/relatorio.html.backup
        else
            echo "⚠️ Nenhuma alteração foi feita nos caminhos"
            echo "💡 Verifique se os caminhos já estão corretos ou execute:"
            echo "   ./debug_image_paths.sh"
            rm docs/relatorio.html.backup
        fi
    fi
fi

# Criar página principal (index.html)
echo "🏠 Criando página principal..."
cat > docs/index.html << 'EOF'
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trabalho 2 RNP - Predição Mackey-Glass</title>
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
        <h1>🧠 Trabalho 2 - Redes Neurais Profundas</h1>
        <h2 style="text-align: center; color: #7f8c8d; margin-bottom: 30px;">Predição de Séries Temporais Mackey-Glass</h2>
        
        <div class="card">
            <h3>📊 Sobre o Projeto</h3>
            <p>Este projeto implementa e compara <span class="highlight">três tipos de redes neurais</span> (MLP, LSTM, GRU) para predição da série temporal Mackey-Glass, com múltiplas configurações e variações para análise abrangente.</p>
            
            <div class="results-grid">
                <div class="metric-card">
                    <h4>🏆 Melhor Modelo</h4>
                    <div class="metric-value">LSTM Bidirectional</div>
                    <p>R² = 0.990789</p>
                </div>
                <div class="metric-card">
                    <h4>🔬 Modelos Avaliados</h4>
                    <div class="metric-value">7</div>
                    <p>Configurações Otimizadas</p>
                </div>
                <div class="metric-card">
                    <h4>📈 Dataset</h4>
                    <div class="metric-value">998</div>
                    <p>Pontos de Validação</p>
                </div>
            </div>
        </div>

        <div class="card">
            <h3>🌐 Relatório Interativo</h3>
            <p>Acesse o relatório completo com <strong>visualizações interativas</strong>, métricas detalhadas (EQMN1, EQMN2, R², RMSE, MAE, MAPE) e análises estatísticas dos experimentos realizados.</p>
            <div style="text-align: center; margin-top: 20px;">
                <a href="relatorio.html" class="btn btn-success">📊 Ver Relatório Interativo Completo</a>
            </div>
        </div>

        <div class="card">
            <h3>📚 Documentação</h3>
            <p>Explore a documentação completa do projeto:</p>
            <a href="https://github.com/ratacheski/Mackey-Glass-Predicit/blob/master/README.md" class="btn">📖 README</a>
            <a href="https://github.com/ratacheski/Mackey-Glass-Predicit/blob/master/COMO_USAR.md" class="btn">🚀 Como Usar</a>
            <a href="https://github.com/ratacheski/Mackey-Glass-Predicit/blob/master/RESULTADOS_FINAIS.md" class="btn">📈 Resultados</a>
            <a href="https://github.com/ratacheski/Mackey-Glass-Predicit/blob/master/RESUMO_EXECUTIVO.md" class="btn">📊 Resumo</a>
        </div>

        <div class="card">
            <h3>📁 Código Fonte</h3>
            <p>Acesse o código completo e reproduzível no GitHub:</p>
            <a href="https://github.com/ratacheski/Mackey-Glass-Predicit" class="btn">💻 Repositório GitHub</a>
        </div>

        <div class="author-info">
            <h3>👨‍🎓 Autor</h3>
            <p><strong>Rafael Ratacheski de Sousa Raulino</strong></p>
            <p>Mestrando em Engenharia Elétrica e de Computação - UFG</p>
            <p>Disciplina: Redes Neurais Profundas - 2025/1</p>
            <p>Data: Junho de 2025</p>
        </div>

        <div class="footer">
            <p>🌐 Hospedado no GitHub Pages | 🔬 Experimentos realizados com PyTorch</p>
        </div>
    </div>
</body>
</html>
EOF

# Solicitar informações do usuário
echo ""
echo "🔧 Configuração personalizada:"

echo "✅ Página principal criada"

# Criar arquivo CSS personalizado
echo "🎨 Criando estilos personalizados..."
cat > docs/css/styles.css << 'EOF'
/* Estilos adicionais para GitHub Pages */
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

# Verificar arquivos criados
echo ""
echo "📊 Verificando arquivos criados:"
echo "📁 docs/"
echo "   ├── index.html ✅"
echo "   ├── relatorio.html $([ -f docs/relatorio.html ] && echo "✅" || echo "❌")"
echo "   ├── css/styles.css ✅"
echo "   └── images/ $([ -d docs/images ] && echo "✅ ($(ls docs/images/ 2>/dev/null | wc -l) arquivos)" || echo "❌")"

# Contar arquivos de imagem
IMG_COUNT=$(find docs/images -type f 2>/dev/null | wc -l)
echo "🖼️ Total de arquivos copiados: $IMG_COUNT"

echo ""
echo "✅ Configuração do GitHub Pages concluída!"
echo ""
echo "📋 PRÓXIMOS PASSOS:"
echo "   1. git add docs/"
echo "   2. git commit -m 'Add GitHub Pages configuration with interactive report'"
echo "   3. git push origin master"
echo "   4. Ir para Settings → Pages no GitHub"
echo "   5. Configurar Source: 'Deploy from a branch'"
echo "   6. Branch: 'master', Folder: '/docs'"
echo "   7. Aguardar deploy (2-3 minutos)"
echo ""
echo "🌐 URL final será:"
echo "   https://ratacheski.github.io/Mackey-Glass-Predicit/"
echo ""
echo "📊 Relatório interativo em:"
echo "   https://ratacheski.github.io/Mackey-Glass-Predicit/relatorio.html"
echo ""
echo "💡 DICA: Adicione este badge ao seu README.md:"
echo "[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live%20Demo-brightgreen?style=for-the-badge&logo=github)](https://ratacheski.github.io/Mackey-Glass-Predicit/)" 