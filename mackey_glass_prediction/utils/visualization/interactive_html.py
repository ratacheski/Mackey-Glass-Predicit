"""
Módulo para geração de relatórios HTML interativos e didáticos
"""
import os
import numpy as np
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def get_html_styles():
    """Retorna os estilos CSS para o relatório HTML"""
    return """
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            box-shadow: 0 0 30px rgba(0,0,0,0.2);
            border-radius: 15px;
            overflow: hidden;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
            position: relative;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .timestamp {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .author-info {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            backdrop-filter: blur(10px);
        }
        
        .author-info h3 {
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        
        .author-info .info-row {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 10px;
        }
        
        .author-info .info-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .content {
            padding: 30px;
        }
        
        .navigation {
            background-color: #34495e;
            padding: 15px;
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 10px;
        }
        
        .nav-btn {
            color: white;
            text-decoration: none;
            padding: 10px 20px;
            border-radius: 25px;
            background-color: rgba(255,255,255,0.1);
            transition: all 0.3s ease;
            font-weight: 500;
        }
        
        .nav-btn:hover {
            background-color: rgba(255,255,255,0.2);
            transform: translateY(-2px);
        }
        
        .nav-btn.active {
            background-color: #3498db;
        }
        
        .section {
            margin: 30px 0;
            opacity: 0;
            transform: translateY(20px);
            transition: all 0.5s ease;
        }
        
        .section.visible {
            opacity: 1;
            transform: translateY(0);
        }
        
        .section h2 {
            color: #2c3e50;
            margin-bottom: 20px;
            border-left: 5px solid #3498db;
            padding-left: 15px;
            font-size: 1.8em;
        }
        
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0,0,0,0.15);
        }
        
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #3498db, #2ecc71);
        }
        
        .metric-title {
            font-size: 1.1em;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .metric-explanation {
            font-size: 0.9em;
            color: #666;
            line-height: 1.4;
        }
        
        .metric-status {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 600;
            margin-top: 10px;
        }
        
        .status-excellent {
            background-color: #d4edda;
            color: #155724;
        }
        
        .status-good {
            background-color: #d1ecf1;
            color: #0c5460;
        }
        
        .status-warning {
            background-color: #fff3cd;
            color: #856404;
        }
        
        .status-poor {
            background-color: #f8d7da;
            color: #721c24;
        }
        
        .model-comparison {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .model-header {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .model-rank {
            background: linear-gradient(135deg, #ffd700, #ffed4a);
            color: #333;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 1.2em;
        }
        
        .model-name {
            font-size: 1.5em;
            font-weight: 600;
            color: #2c3e50;
        }
        
        .toggle-btn {
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .toggle-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
        }
        
        .explanation-panel {
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 20px;
            margin: 20px 0;
            border-radius: 0 10px 10px 0;
            display: none;
        }
        
        .explanation-panel.visible {
            display: block;
            animation: slideDown 0.3s ease;
        }
        
        @keyframes slideDown {
            from {
                opacity: 0;
                transform: translateY(-10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .chart-container {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            margin: 20px 0;
            text-align: center;
        }
        
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            cursor: pointer;
            transition: transform 0.3s ease;
        }
        
        .chart-container img:hover {
            transform: scale(1.02);
        }
        
        .model-charts-section {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .model-charts-header {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #dee2e6;
        }
        
        .model-charts-title {
            font-size: 1.4em;
            font-weight: 600;
            color: #2c3e50;
        }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        /* Modal para tela cheia */
        .modal {
            display: none;
            position: fixed;
            z-index: 10000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
            animation: fadeIn 0.3s ease;
        }
        
        .modal-content {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            max-width: 95%;
            max-height: 95%;
        }
        
        .modal-image {
            width: 100%;
            height: auto;
            border-radius: 10px;
        }
        
        .close-modal {
            position: absolute;
            top: 20px;
            right: 35px;
            color: #fff;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
            z-index: 10001;
        }
        
        .close-modal:hover {
            color: #ccc;
        }
        
        .tooltip {
            position: relative;
            cursor: help;
        }
        
        .tooltip::after {
            content: attr(data-tooltip);
            position: absolute;
            background-color: #333;
            color: white;
            padding: 8px 12px;
            border-radius: 5px;
            font-size: 0.8em;
            white-space: nowrap;
            z-index: 1000;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s ease;
        }
        
        .tooltip:hover::after {
            opacity: 1;
            visibility: visible;
        }
        
        .tabs {
            display: flex;
            border-bottom: 2px solid #e9ecef;
            margin-bottom: 20px;
        }
        
        .tab {
            padding: 15px 25px;
            cursor: pointer;
            background-color: transparent;
            border: none;
            font-size: 1em;
            font-weight: 500;
            color: #6c757d;
            transition: all 0.3s ease;
        }
        
        .tab.active {
            color: #3498db;
            border-bottom: 3px solid #3498db;
            background-color: rgba(52, 152, 219, 0.05);
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .highlight-box {
            background: linear-gradient(135deg, #fff3cd, #ffeaa7);
            border-left: 5px solid #f39c12;
            padding: 20px;
            border-radius: 0 10px 10px 0;
            margin: 20px 0;
        }
        
        .info-icon {
            color: #3498db;
            margin-right: 8px;
        }
        
        .warning-icon {
            color: #f39c12;
            margin-right: 8px;
        }
        
        .success-icon {
            color: #27ae60;
            margin-right: 8px;
        }
        
        @media (max-width: 768px) {
            .metric-grid {
                grid-template-columns: 1fr;
            }
            
            .charts-grid {
                grid-template-columns: 1fr;
            }
            
            .navigation {
                flex-direction: column;
                align-items: center;
            }
            
            .tabs {
                flex-wrap: wrap;
            }
            
            .container {
                margin: 10px;
            }
            
            .content {
                padding: 20px;
            }
            
            .author-info .info-row {
                flex-direction: column;
                gap: 10px;
            }
        }
    </style>
    """

def get_html_scripts():
    """Retorna os scripts JavaScript para interatividade"""
    return """
    <script>
        // Navegação suave entre seções
        function showSection(sectionId) {
            // Esconder todas as seções
            const sections = document.querySelectorAll('.section');
            sections.forEach(section => {
                section.style.display = 'none';
                section.classList.remove('visible');
            });
            
            // Mostrar seção selecionada
            const targetSection = document.getElementById(sectionId);
            if (targetSection) {
                targetSection.style.display = 'block';
                setTimeout(() => {
                    targetSection.classList.add('visible');
                }, 50);
            }
            
            // Atualizar navegação ativa
            const navBtns = document.querySelectorAll('.nav-btn');
            navBtns.forEach(btn => btn.classList.remove('active'));
            
            const activeBtn = document.querySelector(`[onclick="showSection('${sectionId}')"]`);
            if (activeBtn) {
                activeBtn.classList.add('active');
            }
        }
        
        // Mostrar/esconder explicações
        function toggleExplanation(elementId) {
            const panel = document.getElementById(elementId);
            if (panel) {
                if (panel.classList.contains('visible')) {
                    panel.classList.remove('visible');
                    panel.style.display = 'none';
                } else {
                    panel.style.display = 'block';
                    setTimeout(() => {
                        panel.classList.add('visible');
                    }, 50);
                }
            }
        }
        
        // Sistema de abas
        function showTab(tabName) {
            // Esconder todos os conteúdos de aba
            const tabContents = document.querySelectorAll('.tab-content');
            tabContents.forEach(content => {
                content.classList.remove('active');
            });
            
            // Remover classe ativa de todas as abas
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Mostrar conteúdo da aba selecionada
            const targetContent = document.getElementById(tabName);
            if (targetContent) {
                targetContent.classList.add('active');
            }
            
            // Ativar aba selecionada
            const activeTab = document.querySelector(`[onclick="showTab('${tabName}')"]`);
            if (activeTab) {
                activeTab.classList.add('active');
            }
        }
        
        // Modal para visualização em tela cheia
        function openModal(imageSrc, imageTitle) {
            const modal = document.getElementById('imageModal');
            const modalImage = document.getElementById('modalImage');
            
            modal.style.display = 'block';
            modalImage.src = imageSrc;
            modalImage.alt = imageTitle;
            
            // Adicionar título se necessário
            if (!document.querySelector('.modal-title')) {
                const title = document.createElement('h3');
                title.className = 'modal-title';
                title.style.cssText = 'color: white; text-align: center; margin-bottom: 20px; position: absolute; top: 10px; left: 50%; transform: translateX(-50%); z-index: 10002;';
                title.textContent = imageTitle;
                document.querySelector('.modal-content').appendChild(title);
            } else {
                document.querySelector('.modal-title').textContent = imageTitle;
            }
            
            // Prevenir scroll do body
            document.body.style.overflow = 'hidden';
        }
        
        function closeModal() {
            const modal = document.getElementById('imageModal');
            modal.style.display = 'none';
            
            // Restaurar scroll do body
            document.body.style.overflow = 'auto';
        }
        
        // Fechar modal ao clicar fora da imagem
        function setupModalEvents() {
            const modal = document.getElementById('imageModal');
            if (modal) {
                modal.addEventListener('click', function(event) {
                    if (event.target === modal) {
                        closeModal();
                    }
                });
                
                // Fechar modal com tecla ESC
                document.addEventListener('keydown', function(event) {
                    if (event.key === 'Escape') {
                        closeModal();
                    }
                });
            }
        }
        
        // Animações ao fazer scroll
        function handleScrollAnimations() {
            const cards = document.querySelectorAll('.metric-card');
            const windowHeight = window.innerHeight;
            
            cards.forEach(card => {
                const cardTop = card.getBoundingClientRect().top;
                if (cardTop < windowHeight * 0.8) {
                    card.style.opacity = '1';
                    card.style.transform = 'translateY(0)';
                }
            });
        }
        
        // Inicialização quando a página carrega
        document.addEventListener('DOMContentLoaded', function() {
            // Mostrar primeira seção por padrão
            showSection('resumo');
            
            // Configurar modal
            setupModalEvents();
            
            // Configurar animações de scroll
            window.addEventListener('scroll', handleScrollAnimations);
            handleScrollAnimations(); // Executar uma vez ao carregar
            
            // Mostrar primeira aba por padrão em seções com abas
            const firstTab = document.querySelector('.tab');
            if (firstTab) {
                const tabName = firstTab.getAttribute('onclick').match(/'([^']+)'/)[1];
                showTab(tabName);
            }
        });
        
        // Adicionar efeitos de hover para cards
        document.addEventListener('DOMContentLoaded', function() {
            const cards = document.querySelectorAll('.metric-card');
            cards.forEach(card => {
                card.addEventListener('mouseenter', function() {
                    this.style.transform = 'translateY(-5px) scale(1.02)';
                });
                
                card.addEventListener('mouseleave', function() {
                    this.style.transform = 'translateY(0) scale(1)';
                });
            });
            
            // Efeitos para imagens de gráficos
            const images = document.querySelectorAll('.chart-container img');
            images.forEach(img => {
                img.addEventListener('mouseenter', function() {
                    this.style.transform = 'scale(1.05)';
                    this.style.boxShadow = '0 8px 25px rgba(0,0,0,0.2)';
                });
                
                img.addEventListener('mouseleave', function() {
                    this.style.transform = 'scale(1)';
                    this.style.boxShadow = '0 3px 10px rgba(0,0,0,0.1)';
                });
            });
        });
        
        // Função para destacar métricas importantes
        function highlightBestMetrics() {
            const metricCards = document.querySelectorAll('.metric-card');
            metricCards.forEach(card => {
                const value = card.querySelector('.metric-value');
                const title = card.querySelector('.metric-title');
                
                if (title && title.textContent.includes('R²')) {
                    const r2Value = parseFloat(value.textContent);
                    if (r2Value > 0.95) {
                        card.style.border = '2px solid #27ae60';
                        card.style.boxShadow = '0 0 20px rgba(39, 174, 96, 0.3)';
                    }
                }
            });
        }
        
        // Executar após carregamento completo
        window.addEventListener('load', function() {
            highlightBestMetrics();
            console.log('🎉 Relatório HTML Interativo totalmente carregado!');
        });
    </script>
    """

def get_metric_explanations():
    """Retorna explicações detalhadas das métricas"""
    return {
        'r2': {
            'name': 'R² (Coeficiente de Determinação)',
            'icon': '📊',
            'explanation': '''
            O R² mede o quão bem o modelo explica a variabilidade dos dados.
            <br><br>
            <strong>Interpretação:</strong>
            <ul>
                <li><strong>R² = 1.0:</strong> Predição perfeita</li>
                <li><strong>R² > 0.9:</strong> Excelente</li>
                <li><strong>R² > 0.8:</strong> Bom</li>
                <li><strong>R² > 0.6:</strong> Moderado</li>
                <li><strong>R² < 0.6:</strong> Baixo</li>
            </ul>
            <br>
            <strong>Fórmula:</strong> R² = 1 - (SSres / SStot)
            ''',
            'ranges': {
                'excellent': 0.9,
                'good': 0.8,
                'moderate': 0.6
            }
        },
        'rmse': {
            'name': 'RMSE (Raiz do Erro Quadrático Médio)',
            'icon': '📏',
            'explanation': '''
            O RMSE mede a magnitude típica dos erros de predição.
            <br><br>
            <strong>Características:</strong>
            <ul>
                <li>Unidade igual aos dados originais</li>
                <li>Penaliza erros grandes mais severamente</li>
                <li>Valores menores indicam melhor performance</li>
                <li>Sensível a outliers</li>
            </ul>
            <br>
            <strong>Fórmula:</strong> RMSE = √(Σ(yi - ŷi)² / n)
            '''
        },
        'mae': {
            'name': 'MAE (Erro Absoluto Médio)',
            'icon': '📐',
            'explanation': '''
            O MAE mede o erro médio absoluto entre predições e valores reais.
            <br><br>
            <strong>Características:</strong>
            <ul>
                <li>Mais robusto a outliers que RMSE</li>
                <li>Interpretação intuitiva</li>
                <li>Unidade igual aos dados originais</li>
                <li>Sempre não-negativo</li>
            </ul>
            <br>
            <strong>Fórmula:</strong> MAE = Σ|yi - ŷi| / n
            '''
        },
        'mse': {
            'name': 'MSE (Erro Quadrático Médio)',
            'icon': '📈',
            'explanation': '''
            O MSE mede a média dos quadrados dos erros de predição.
            <br><br>
            <strong>Características:</strong>
            <ul>
                <li>Base para cálculo do RMSE</li>
                <li>Penaliza erros grandes quadraticamente</li>
                <li>Sempre não-negativo</li>
                <li>Unidade quadrática dos dados originais</li>
            </ul>
            <br>
            <strong>Fórmula:</strong> MSE = Σ(yi - ŷi)² / n
            '''
        },
        'mape': {
            'name': 'MAPE (Erro Percentual Absoluto Médio)',
            'icon': '📊',
            'explanation': '''
            O MAPE expressa o erro como percentual dos valores reais.
            <br><br>
            <strong>Interpretação:</strong>
            <ul>
                <li><strong>MAPE < 5%:</strong> Excelente precisão</li>
                <li><strong>MAPE < 10%:</strong> Boa precisão</li>
                <li><strong>MAPE < 20%:</strong> Precisão moderada</li>
                <li><strong>MAPE ≥ 20%:</strong> Baixa precisão</li>
            </ul>
            <br>
            <strong>Fórmula:</strong> MAPE = (100/n) × Σ|yi - ŷi|/|yi|
            '''
        },
        'eqmn1': {
            'name': 'EQMN1 (Erro Quadrático Médio Normalizado - Variância)',
            'icon': '📊',
            'explanation': '''
            O EQMN1 normaliza o MSE pela variância dos valores reais.
            <br><br>
            <strong>Características:</strong>
            <ul>
                <li>Normalizado entre 0 e 1</li>
                <li>Independente da escala dos dados</li>
                <li>Útil para comparação entre datasets</li>
                <li>Valores menores indicam melhor performance</li>
            </ul>
            <br>
            <strong>Fórmula:</strong> EQMN1 = MSE / Var(y_true)
            '''
        },
        'eqmn2': {
            'name': 'EQMN2 (Erro Quadrático Médio Normalizado - Modelo Naive)',
            'icon': '📉',
            'explanation': '''
            O EQMN2 normaliza o MSE pelo MSE de um modelo naive (persistence).
            <br><br>
            <strong>Características:</strong>
            <ul>
                <li>Compara com modelo naive que usa valor anterior</li>
                <li>Ideal para séries temporais</li>
                <li>Valores < 1 indicam que o modelo supera a persistência</li>
                <li>Valores > 1 indicam que o modelo é pior que usar valor anterior</li>
            </ul>
            <br>
            <strong>Fórmula:</strong> EQMN2 = MSE / MSE_naive
            <br>
            <strong>Interpretação:</strong>
            <ul>
                <li><strong>< 0.5:</strong> Muito superior ao modelo naive</li>
                <li><strong>0.5-1.0:</strong> Superior ao modelo naive</li>
                <li><strong>= 1.0:</strong> Equivalente ao modelo naive</li>
                <li><strong>> 1.0:</strong> Inferior ao modelo naive</li>
            </ul>
            '''
        }
    }

def calculate_metrics(actuals, predictions):
    """Calcula todas as métricas para um modelo"""
    actuals = np.array(actuals).flatten()
    predictions = np.array(predictions).flatten()
    
    r2 = r2_score(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100 if np.all(actuals != 0) else np.nan
    
    # Calcular EQMN1 (MSE normalizado pela variância)
    var_actuals = np.var(actuals)
    eqmn1 = mse / var_actuals if var_actuals != 0 else np.nan
    
    # Calcular EQMN2 (MSE normalizado pelo MSE naive/persistence)
    if len(actuals) > 1:
        # Modelo naive: usar valor anterior como predição
        x_pa = np.roll(actuals, 1)
        x_pa[0] = actuals[0]  # Primeiro valor permanece igual
        
        naive_mse = np.mean((x_pa - actuals) ** 2)
        eqmn2 = mse / naive_mse if naive_mse != 0 else np.nan
    else:
        eqmn2 = np.nan
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mse': mse,
        'mape': mape,
        'eqmn1': eqmn1,
        'eqmn2': eqmn2
    }

def get_metric_status(metric_name, value):
    """Determina o status de uma métrica"""
    explanations = get_metric_explanations()
    
    if metric_name == 'r2':
        if value >= 0.9:
            return 'status-excellent', 'Excelente'
        elif value >= 0.8:
            return 'status-good', 'Bom'
        elif value >= 0.6:
            return 'status-warning', 'Moderado'
        else:
            return 'status-poor', 'Baixo'
    
    elif metric_name == 'mape' and not np.isnan(value):
        if value < 5:
            return 'status-excellent', 'Excelente'
        elif value < 10:
            return 'status-good', 'Bom'
        elif value < 20:
            return 'status-warning', 'Moderado'
        else:
            return 'status-poor', 'Baixo'
    
    # Para RMSE e MAE, o status depende do contexto relativo
    return 'status-good', 'Normal'

def generate_interactive_html_report(results_dict, generated_files, save_path, report_type="comparison"):
    """
    Gera relatório HTML interativo e didático
    
    Args:
        results_dict: Dicionário com resultados dos modelos
        generated_files: Dicionário com caminhos dos arquivos gerados
        save_path: Caminho para salvar o relatório HTML
        report_type: Tipo do relatório ("single" ou "comparison")
    """
    explanations = get_metric_explanations()
    timestamp = datetime.now().strftime('%d/%m/%Y às %H:%M:%S')
    
    # Cabeçalho HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Relatório Interativo - Análise de Modelos de Machine Learning</title>
        {get_html_styles()}
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 Relatório Interativo de Machine Learning</h1>
                <div class="timestamp">📅 Gerado em: {timestamp}</div>
                
                <div class="author-info">
                    <h3>👨‍🎓 Autor</h3>
                    <div class="info-row">
                        <div class="info-item">
                            <span>👤</span>
                            <strong>Rafael Ratacheski de Sousa Raulino</strong>
                        </div>
                        <div class="info-item">
                            <span>📧</span>
                            <a href="mailto:ratacheski@discente.ufg.br" style="color: white;">ratacheski@discente.ufg.br</a>
                        </div>
                    </div>
                    <div class="info-row">
                        <div class="info-item">
                            <span>🎓</span>
                            Mestrando em Engenharia Elétrica e de Computação
                        </div>
                        <div class="info-item">
                            <span>🏛️</span>
                            PPGEEC - UFG
                        </div>
                    </div>
                    <div class="info-row">
                        <div class="info-item">
                            <span>📚</span>
                            Disciplina: Redes Neurais Profundas
                        </div>
                        <div class="info-item">
                            <span>📅</span>
                            Período: 2025/1
                        </div>
                        <div class="info-item">
                            <span>📝</span>
                            Trabalho Computacional 2
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="navigation">
                <a href="#" class="nav-btn" onclick="showSection('resumo')">📋 Resumo Executivo</a>
                <a href="#" class="nav-btn" onclick="showSection('metricas')">📊 Métricas Detalhadas</a>
                <a href="#" class="nav-btn" onclick="showSection('graficos')">📈 Visualizações</a>
                <a href="#" class="nav-btn" onclick="showSection('analises')">🔬 Análises Estatísticas</a>
                <a href="#" class="nav-btn" onclick="showSection('guia')">📚 Guia de Métricas</a>
            </div>
            
            <div class="content">
    """
    
    # Seção Resumo Executivo
    html_content += f"""
                <div id="resumo" class="section">
                    <h2>📋 Resumo Executivo</h2>
                    
                    <div class="highlight-box">
                        <h3><span class="info-icon">ℹ️</span>Informações Gerais</h3>
                        <p><strong>Tipo de análise:</strong> {'Comparação de múltiplos modelos' if report_type == 'comparison' else 'Análise de modelo único'}</p>
                        <p><strong>Número de modelos:</strong> {len(results_dict)}</p>
                        <p><strong>Modelos analisados:</strong> {', '.join(results_dict.keys())}</p>
                        <p><strong>Data de geração:</strong> {timestamp}</p>
                    </div>
    """
    
    if report_type == "comparison" and len(results_dict) > 1:
        # Ranking dos modelos
        model_rankings = []
        for model_name, results in results_dict.items():
            if 'actuals' in results and 'predictions' in results:
                metrics = calculate_metrics(results['actuals'], results['predictions'])
                score = metrics['r2']  # Usar R² como métrica principal para ranking
                model_rankings.append((model_name, score, metrics))
        
        model_rankings.sort(key=lambda x: x[1], reverse=True)
        
        html_content += """
                    <div class="model-comparison">
                        <h3><span class="success-icon">🏆</span>Ranking dos Modelos</h3>
        """
        
        for rank, (model_name, score, metrics) in enumerate(model_rankings, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}°"
            status_class, status_text = get_metric_status('r2', score)
            
            html_content += f"""
                        <div class="model-header">
                            <div class="model-rank">{medal}</div>
                            <div class="model-name">{model_name}</div>
                            <div class="metric-status {status_class}">{status_text}</div>
                        </div>
            """
    
    html_content += """
                    </div>
                </div>
    """
    
    # Seção Métricas Detalhadas
    html_content += """
                <div id="metricas" class="section">
                    <h2>📊 Métricas Detalhadas</h2>
                    
                    <div class="tabs">
                        <button class="tab active" onclick="showTab('metricas-visao-geral')">Visão Geral</button>
                        <button class="tab" onclick="showTab('metricas-detalhadas')">Análise Detalhada</button>
                    </div>
                    
                    <div id="metricas-visao-geral" class="tab-content active">
    """
    
    # Cards de métricas para cada modelo
    for model_name, results in results_dict.items():
        if 'actuals' in results and 'predictions' in results:
            metrics = calculate_metrics(results['actuals'], results['predictions'])
            
            html_content += f"""
                        <h3>{model_name}</h3>
                        <div class="metric-grid">
            """
            
            # Card R²
            r2_status_class, r2_status_text = get_metric_status('r2', metrics['r2'])
            html_content += f"""
                            <div class="metric-card">
                                <div class="metric-title">
                                    {explanations['r2']['icon']} {explanations['r2']['name']}
                                    <button class="toggle-btn" onclick="toggleExplanation('r2-explanation-{model_name.replace(" ", "_")}')">?</button>
                                </div>
                                <div class="metric-value" style="color: {'#27ae60' if metrics['r2'] > 0.9 else '#f39c12' if metrics['r2'] > 0.6 else '#e74c3c'}">
                                    {metrics['r2']:.6f}
                                </div>
                                <div class="metric-status {r2_status_class}">{r2_status_text}</div>
                                <div id="r2-explanation-{model_name.replace(" ", "_")}" class="explanation-panel">
                                    {explanations['r2']['explanation']}
                                </div>
                            </div>
            """
            
            # Card RMSE
            html_content += f"""
                            <div class="metric-card">
                                <div class="metric-title">
                                    {explanations['rmse']['icon']} {explanations['rmse']['name']}
                                    <button class="toggle-btn" onclick="toggleExplanation('rmse-explanation-{model_name.replace(" ", "_")}')">?</button>
                                </div>
                                <div class="metric-value" style="color: #3498db">
                                    {metrics['rmse']:.6f}
                                </div>
                                <div class="metric-explanation">Menor é melhor</div>
                                <div id="rmse-explanation-{model_name.replace(" ", "_")}" class="explanation-panel">
                                    {explanations['rmse']['explanation']}
                                </div>
                            </div>
            """
            
            # Card MAE
            html_content += f"""
                            <div class="metric-card">
                                <div class="metric-title">
                                    {explanations['mae']['icon']} {explanations['mae']['name']}
                                    <button class="toggle-btn" onclick="toggleExplanation('mae-explanation-{model_name.replace(" ", "_")}')">?</button>
                                </div>
                                <div class="metric-value" style="color: #9b59b6">
                                    {metrics['mae']:.6f}
                                </div>
                                <div class="metric-explanation">Menor é melhor</div>
                                <div id="mae-explanation-{model_name.replace(" ", "_")}" class="explanation-panel">
                                    {explanations['mae']['explanation']}
                                </div>
                            </div>
            """
            
            # Card MSE
            html_content += f"""
                            <div class="metric-card">
                                <div class="metric-title">
                                    {explanations['mse']['icon']} {explanations['mse']['name']}
                                    <button class="toggle-btn" onclick="toggleExplanation('mse-explanation-{model_name.replace(" ", "_")}')">?</button>
                                </div>
                                <div class="metric-value" style="color: #3498db">
                                    {metrics['mse']:.6f}
                                </div>
                                <div class="metric-explanation">Menor é melhor</div>
                                <div id="mse-explanation-{model_name.replace(" ", "_")}" class="explanation-panel">
                                    {explanations['mse']['explanation']}
                                </div>
                            </div>
            """
            
            # Card MAPE (se disponível)
            if not np.isnan(metrics['mape']):
                mape_status_class, mape_status_text = get_metric_status('mape', metrics['mape'])
                html_content += f"""
                            <div class="metric-card">
                                <div class="metric-title">
                                    {explanations['mape']['icon']} {explanations['mape']['name']}
                                    <button class="toggle-btn" onclick="toggleExplanation('mape-explanation-{model_name.replace(" ", "_")}')">?</button>
                                </div>
                                <div class="metric-value" style="color: {'#27ae60' if metrics['mape'] < 5 else '#f39c12' if metrics['mape'] < 20 else '#e74c3c'}">
                                    {metrics['mape']:.2f}%
                                </div>
                                <div class="metric-status {mape_status_class}">{mape_status_text}</div>
                                <div id="mape-explanation-{model_name.replace(" ", "_")}" class="explanation-panel">
                                    {explanations['mape']['explanation']}
                                </div>
                            </div>
                """
            
            # Card EQMN1
            if not np.isnan(metrics['eqmn1']):
                html_content += f"""
                            <div class="metric-card">
                                <div class="metric-title">
                                    {explanations['eqmn1']['icon']} {explanations['eqmn1']['name']}
                                    <button class="toggle-btn" onclick="toggleExplanation('eqmn1-explanation-{model_name.replace(" ", "_")}')">?</button>
                                </div>
                                <div class="metric-value" style="color: {'#27ae60' if metrics['eqmn1'] < 0.1 else '#f39c12' if metrics['eqmn1'] < 0.5 else '#e74c3c'}">
                                    {metrics['eqmn1']:.6f}
                                </div>
                                <div class="metric-explanation">Menor é melhor (normalizado)</div>
                                <div id="eqmn1-explanation-{model_name.replace(" ", "_")}" class="explanation-panel">
                                    {explanations['eqmn1']['explanation']}
                                </div>
                            </div>
                """
            
            # Card EQMN2
            if not np.isnan(metrics['eqmn2']):
                html_content += f"""
                            <div class="metric-card">
                                <div class="metric-title">
                                    {explanations['eqmn2']['icon']} {explanations['eqmn2']['name']}
                                    <button class="toggle-btn" onclick="toggleExplanation('eqmn2-explanation-{model_name.replace(" ", "_")}')">?</button>
                                </div>
                                <div class="metric-value" style="color: {'#27ae60' if metrics['eqmn2'] < 0.5 else '#2ecc71' if metrics['eqmn2'] < 1.0 else '#f39c12' if metrics['eqmn2'] < 2.0 else '#e74c3c'}">
                                    {metrics['eqmn2']:.6f}
                                </div>
                                <div class="metric-explanation">Menor que 1.0 é melhor (supera modelo naive)</div>
                                <div id="eqmn2-explanation-{model_name.replace(" ", "_")}" class="explanation-panel">
                                    {explanations['eqmn2']['explanation']}
                                </div>
                            </div>
                """
            
            html_content += """
                        </div>
            """
    
    html_content += """
                    </div>
                    
                    <div id="metricas-detalhadas" class="tab-content">
                        <div class="highlight-box">
                            <h3><span class="info-icon">📚</span>Interpretação Avançada das Métricas</h3>
                            <p>Esta seção oferece uma análise mais profunda do significado de cada métrica no contexto do seu modelo de machine learning.</p>
                        </div>
                        
                        <h4>🎯 Como Interpretar os Resultados</h4>
                        <ul>
                            <li><strong>R² próximo de 1:</strong> Indica que o modelo explica quase toda a variabilidade dos dados</li>
                            <li><strong>RMSE baixo:</strong> Erros de predição pequenos em relação à escala dos dados</li>
                            <li><strong>MAE baixo:</strong> Erro médio absoluto pequeno, mais robusto a outliers</li>
                            <li><strong>MAPE baixo:</strong> Erro percentual pequeno, útil para comparação entre diferentes escalas</li>
                        </ul>
                    </div>
                </div>
    """
    
    # Seção de Visualizações
    html_content += """
                <div id="graficos" class="section">
                    <h2>📈 Visualizações</h2>
                    
                    <div class="highlight-box">
                        <h3><span class="info-icon">📊</span>Gráficos Explicativos</h3>
                        <p>Os gráficos abaixo mostram diferentes aspectos da performance dos modelos. Clique em cada imagem para visualizar em tela cheia.</p>
                    </div>
                    
                    <div class="tabs">
                        <button class="tab active" onclick="showTab('graficos-comparativos')">📊 Gráficos Comparativos</button>
                        <button class="tab" onclick="showTab('graficos-individuais')">🔍 Análises Individuais</button>
                    </div>
                    
                    <div id="graficos-comparativos" class="tab-content active">
                        <h3>🌐 Análises Comparativas</h3>
                        <div class="metric-grid">
    """
    
    # Adicionar gráficos comparativos
    comparative_charts = ['overview', 'metrics_comparison', 'metrics_table']
    for file_key, file_path in generated_files.items():
        if any(keyword in file_key for keyword in comparative_charts) and file_path.endswith('.png'):
            file_name = os.path.basename(file_path)
            
            title_map = {
                'overview': '🌐 Visão Geral Comparativa',
                'metrics_comparison': '📊 Comparação de Métricas',
                'metrics_table': '📋 Tabela de Métricas'
            }
            
            title = title_map.get(file_key.split('_')[0], file_name.replace('_', ' ').replace('.png', '').title())
            
            html_content += f"""
                            <div class="chart-container">
                                <h4>{title}</h4>
                                <img src="{file_name}" alt="{title}" onclick="openModal('{file_name}', '{title}')">
                            </div>
            """
    
    html_content += """
                        </div>
                    </div>
                    
                    <div id="graficos-individuais" class="tab-content">
                        <h3>🔍 Análises por Modelo</h3>
    """
    
    # Organizar gráficos por modelo
    models_in_files = set()
    for file_key in generated_files.keys():
        for model_name in results_dict.keys():
            clean_model_name = model_name.replace(" ", "_")
            if clean_model_name in file_key:
                models_in_files.add(model_name)
                break
    
    for model_name in models_in_files:
        clean_model_name = model_name.replace(" ", "_")
        model_files = {k: v for k, v in generated_files.items() if clean_model_name in k and v.endswith('.png')}
        
        if model_files:
            html_content += f"""
                        <div class="model-charts-section">
                            <div class="model-charts-header">
                                <h4 class="model-charts-title">🤖 {model_name}</h4>
                            </div>
                            <div class="charts-grid">
            """
            
            for file_key, file_path in model_files.items():
                file_name = os.path.basename(file_path)
                
                # Títulos mais descritivos
                title_map = {
                    'training': '📈 Histórico de Treinamento',
                    'predictions': '🎯 Predições vs Valores Reais',
                    'qq': '📊 Análise Q-Q Plot',
                    'cdf': '📋 Função de Distribuição Acumulada',
                    'pdf': '📊 Função de Densidade de Probabilidade',
                    'ks': '🔬 Teste Kolmogorov-Smirnov',
                    'residuals': '📉 Análise de Resíduos',
                    'autocorr': '🔄 Autocorrelação'
                }
                
                # Identificar tipo do gráfico
                chart_type = 'outros'
                for key in title_map.keys():
                    if key in file_key:
                        chart_type = key
                        break
                
                title = title_map.get(chart_type, file_name.replace('_', ' ').replace('.png', '').title())
                
                html_content += f"""
                                <div class="chart-container">
                                    <h5>{title}</h5>
                                    <img src="{file_name}" alt="{title}" onclick="openModal('{file_name}', '{model_name} - {title}')">
                                </div>
                """
            
            html_content += """
                            </div>
                        </div>
            """
    
    html_content += """
                    </div>
                </div>
    """
    
    # Modal para tela cheia
    html_content += """
                <!-- Modal para visualização em tela cheia -->
                <div id="imageModal" class="modal">
                    <span class="close-modal" onclick="closeModal()">&times;</span>
                    <div class="modal-content">
                        <img id="modalImage" class="modal-image" alt="Imagem em tela cheia">
                    </div>
                </div>
    """
    
    # Seção de Análises Estatísticas
    html_content += """
                <div id="analises" class="section">
                    <h2>🔬 Análises Estatísticas</h2>
                    
                    <div class="highlight-box">
                        <h3><span class="info-icon">🧪</span>Testes e Análises Avançadas</h3>
                        <p>Esta seção apresenta análises estatísticas aprofundadas dos resultados do modelo.</p>
                    </div>
                    
                    <h3>📊 Análises Disponíveis</h3>
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-title">📊 Q-Q Plot</div>
                            <div class="metric-explanation">
                                Compara a distribuição dos resíduos com a distribuição normal. 
                                Pontos próximos à linha diagonal indicam normalidade dos resíduos.
                            </div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-title">📈 Análise de Distribuições</div>
                            <div class="metric-explanation">
                                Compara as funções de distribuição (CDF) e densidade (PDF) entre 
                                valores reais e predições.
                            </div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-title">🔬 Teste Kolmogorov-Smirnov</div>
                            <div class="metric-explanation">
                                Testa se as distribuições de valores reais e predições são 
                                estatisticamente similares.
                            </div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-title">🔄 Análise de Autocorrelação</div>
                            <div class="metric-explanation">
                                Examina a dependência temporal nos dados, importante para 
                                séries temporais.
                            </div>
                        </div>
                    </div>
                </div>
    """
    
    # Seção Guia de Métricas
    html_content += """
                <div id="guia" class="section">
                    <h2>📚 Guia Completo de Métricas</h2>
                    
                    <div class="highlight-box">
                        <h3><span class="info-icon">🎓</span>Aprenda sobre Métricas de Machine Learning</h3>
                        <p>Este guia explica em detalhes todas as métricas utilizadas na avaliação dos modelos.</p>
                    </div>
                    
                    <div class="metric-grid">
    """
    
    # Cards explicativos para cada métrica
    for metric_key, metric_info in explanations.items():
        html_content += f"""
                        <div class="metric-card">
                            <div class="metric-title">
                                {metric_info['icon']} {metric_info['name']}
                            </div>
                            <div class="metric-explanation">
                                {metric_info['explanation']}
                            </div>
                        </div>
        """
    
    html_content += """
                    </div>
                    
                    <div class="highlight-box">
                        <h3><span class="success-icon">💡</span>Dicas de Interpretação</h3>
                        <ul>
                            <li><strong>Use múltiplas métricas:</strong> Cada métrica oferece uma perspectiva diferente</li>
                            <li><strong>Considere o contexto:</strong> A importância de cada métrica depende da aplicação</li>
                            <li><strong>Analise os resíduos:</strong> Padrões nos erros podem revelar problemas no modelo</li>
                            <li><strong>Compare com baselines:</strong> Avalie se o modelo supera métodos simples</li>
                            <li><strong>Valide em dados novos:</strong> Performance em teste é crucial</li>
                        </ul>
                    </div>
                </div>
    """
    
    # Fechar HTML
    html_content += f"""
            </div>
        </div>
        
        {get_html_scripts()}
        
        <script>
            // Adicionar funcionalidade específica para este relatório
            document.addEventListener('DOMContentLoaded', function() {{
                console.log('Relatório HTML Interativo carregado!');
                
                // Adicionar contador de cliques em explicações
                let explanationClicks = 0;
                document.querySelectorAll('.toggle-btn').forEach(btn => {{
                    btn.addEventListener('click', function() {{
                        explanationClicks++;
                        console.log(`Explicação visualizada ${{explanationClicks}} vezes`);
                    }});
                }});
            }});
        </script>
    </body>
    </html>
    """
    
    # Salvar arquivo
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Relatório HTML interativo gerado: {save_path}")
    return save_path 