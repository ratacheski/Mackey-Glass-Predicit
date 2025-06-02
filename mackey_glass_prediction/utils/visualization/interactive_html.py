"""
Interactive HTML report generation module
"""
import os
import numpy as np
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def get_html_styles():
    """Returns CSS styles for the HTML report"""
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
            showSection('summary');
            
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
            console.log('🎉 Interactive HTML Report fully loaded!');
        });
    </script>
    """

def get_metric_explanations():
    """Returns detailed metric explanations"""
    return {
        'r2': {
            'name': 'R² (Coefficient of Determination)',
            'icon': '📊',
            'explanation': '''
            R² measures how well the model explains the data variability.
            <br><br>
            <strong>Interpretation:</strong>
            <ul>
                <li><strong>R² = 1.0:</strong> Perfect prediction</li>
                <li><strong>R² > 0.9:</strong> Excellent</li>
                <li><strong>R² > 0.8:</strong> Good</li>
                <li><strong>R² > 0.6:</strong> Moderate</li>
                <li><strong>R² < 0.6:</strong> Low</li>
            </ul>
            <br>
            <strong>Formula:</strong> R² = 1 - (SSres / SStot)
            ''',
            'ranges': {
                'excellent': 0.9,
                'good': 0.8,
                'moderate': 0.6
            }
        },
        'rmse': {
            'name': 'RMSE (Root Mean Square Error)',
            'icon': '📏',
            'explanation': '''
            RMSE measures the typical magnitude of prediction errors.
            <br><br>
            <strong>Characteristics:</strong>
            <ul>
                <li>Same unit as original data</li>
                <li>Penalizes large errors more severely</li>
                <li>Lower values indicate better performance</li>
                <li>Sensitive to outliers</li>
            </ul>
            <br>
            <strong>Formula:</strong> RMSE = √(Σ(yi - ŷi)² / n)
            '''
        },
        'mae': {
            'name': 'MAE (Mean Absolute Error)',
            'icon': '📐',
            'explanation': '''
            MAE measures the mean absolute error between predictions and actual values.
            <br><br>
            <strong>Characteristics:</strong>
            <ul>
                <li>More robust to outliers than RMSE</li>
                <li>Intuitive interpretation</li>
                <li>Same unit as original data</li>
                <li>Always non-negative</li>
            </ul>
            <br>
            <strong>Formula:</strong> MAE = Σ|yi - ŷi| / n
            '''
        },
        'mse': {
            'name': 'MSE (Mean Square Error)',
            'icon': '📈',
            'explanation': '''
            MSE measures the mean of squared prediction errors.
            <br><br>
            <strong>Characteristics:</strong>
            <ul>
                <li>Base for RMSE calculation</li>
                <li>Penalizes large errors quadratically</li>
                <li>Always non-negative</li>
                <li>Squared unit of original data</li>
            </ul>
            <br>
            <strong>Formula:</strong> MSE = Σ(yi - ŷi)² / n
            '''
        },
        'mape': {
            'name': 'MAPE (Mean Absolute Percentage Error)',
            'icon': '📊',
            'explanation': '''
            MAPE expresses error as a percentage of actual values.
            <br><br>
            <strong>Interpretation:</strong>
            <ul>
                <li><strong>MAPE < 5%:</strong> Excellent accuracy</li>
                <li><strong>MAPE < 10%:</strong> Good accuracy</li>
                <li><strong>MAPE < 20%:</strong> Moderate accuracy</li>
                <li><strong>MAPE ≥ 20%:</strong> Low accuracy</li>
            </ul>
            <br>
            <strong>Formula:</strong> MAPE = (100/n) × Σ|yi - ŷi|/|yi|
            '''
        },
        'eqmn1': {
            'name': 'NMSE1 (Normalized Mean Square Error - Variance)',
            'icon': '📊',
            'explanation': '''
            NMSE1 normalizes MSE by the variance of actual values.
            <br><br>
            <strong>Characteristics:</strong>
            <ul>
                <li>Normalized between 0 and 1</li>
                <li>Scale-independent</li>
                <li>Useful for comparison across datasets</li>
                <li>Lower values indicate better performance</li>
            </ul>
            <br>
            <strong>Formula:</strong> NMSE1 = MSE / Var(y_true)
            '''
        },
        'eqmn2': {
            'name': 'NMSE2 (Normalized Mean Square Error - Naive Model)',
            'icon': '📉',
            'explanation': '''
            NMSE2 normalizes MSE by the MSE of a naive model (persistence).
            <br><br>
            <strong>Characteristics:</strong>
            <ul>
                <li>Compares with naive model using previous value</li>
                <li>Ideal for time series</li>
                <li>Values < 1 indicate model outperforms persistence</li>
                <li>Values > 1 indicate model performs worse than using previous value</li>
            </ul>
            <br>
            <strong>Formula:</strong> NMSE2 = MSE / MSE_naive
            <br>
            <strong>Interpretation:</strong>
            <ul>
                <li><strong>< 0.5:</strong> Much superior to naive model</li>
                <li><strong>0.5-1.0:</strong> Superior to naive model</li>
                <li><strong>= 1.0:</strong> Equivalent to naive model</li>
                <li><strong>> 1.0:</strong> Inferior to naive model</li>
            </ul>
            '''
        }
    }

def calculate_metrics(actuals, predictions):
    """Calculate all metrics for a model"""
    actuals = np.array(actuals).flatten()
    predictions = np.array(predictions).flatten()
    
    r2 = r2_score(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100 if np.all(actuals != 0) else np.nan
    
    # Calculate EQMN1 (MSE normalized by variance)
    var_actuals = np.var(actuals)
    eqmn1 = mse / var_actuals if var_actuals != 0 else np.nan
    
    # Calculate EQMN2 (MSE normalized by naive/persistence MSE)
    if len(actuals) > 1:
        # Naive model: use previous value as prediction
        x_pa = np.roll(actuals, 1)
        x_pa[0] = actuals[0]  # First value remains the same
        
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
    """Determine the status of a metric"""
    explanations = get_metric_explanations()
    
    if metric_name == 'r2':
        if value >= 0.9:
            return 'status-excellent', 'Excellent'
        elif value >= 0.8:
            return 'status-good', 'Good'
        elif value >= 0.6:
            return 'status-warning', 'Moderate'
        else:
            return 'status-poor', 'Low'
    
    elif metric_name == 'mape' and not np.isnan(value):
        if value < 5:
            return 'status-excellent', 'Excellent'
        elif value < 10:
            return 'status-good', 'Good'
        elif value < 20:
            return 'status-warning', 'Moderate'
        else:
            return 'status-poor', 'Low'
    
    # For RMSE and MAE, status depends on relative context
    return 'status-good', 'Normal'

def generate_interactive_html_report(results_dict, generated_files, save_path, report_type="comparison"):
    """
    Generate interactive and didactic HTML report
    
    Args:
        results_dict: Dictionary with model results
        generated_files: Dictionary with generated file paths
        save_path: Path to save the HTML report
        report_type: Report type ("single" or "comparison")
    """
    explanations = get_metric_explanations()
    timestamp = datetime.now().strftime('%d/%m/%Y at %H:%M:%S')
    
    # HTML header
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Interactive Report - Machine Learning Model Analysis</title>
        {get_html_styles()}
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 Interactive Machine Learning Report</h1>
                <div class="timestamp">📅 Generated on: {timestamp}</div>
                
                <div class="author-info">
                    <h3>👨‍🎓 Author</h3>
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
                            MSc Student in Electrical and Computer Engineering
                        </div>
                        <div class="info-item">
                            <span>🏛️</span>
                            PPGEEC - UFG
                        </div>
                    </div>
                    <div class="info-row">
                        <div class="info-item">
                            <span>📚</span>
                            Course: Deep Neural Networks
                        </div>
                        <div class="info-item">
                            <span>📅</span>
                            Period: 2025/1
                        </div>
                        <div class="info-item">
                            <span>📝</span>
                            Computational Assignment 2
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="navigation">
                <a href="#" class="nav-btn" onclick="showSection('summary')">📋 Executive Summary</a>
                <a href="#" class="nav-btn" onclick="showSection('metrics')">📊 Detailed Metrics</a>
                <a href="#" class="nav-btn" onclick="showSection('charts')">📈 Visualizations</a>
                <a href="#" class="nav-btn" onclick="showSection('analyses')">🔬 Statistical Analyses</a>
                <a href="#" class="nav-btn" onclick="showSection('guide')">📚 Metrics Guide</a>
            </div>
            
            <div class="content">
    """
    
    # Executive Summary Section
    html_content += f"""
                <div id="summary" class="section">
                    <h2>📋 Executive Summary</h2>
                    
                    <div class="highlight-box">
                        <h3><span class="info-icon">ℹ️</span>General Information</h3>
                        <p><strong>Analysis type:</strong> {'Multi-model comparison' if report_type == 'comparison' else 'Single model analysis'}</p>
                        <p><strong>Number of models:</strong> {len(results_dict)}</p>
                        <p><strong>Models analyzed:</strong> {', '.join(results_dict.keys())}</p>
                        <p><strong>Generation date:</strong> {timestamp}</p>
                    </div>
    """
    
    if report_type == "comparison" and len(results_dict) > 1:
        # Model ranking
        model_rankings = []
        for model_name, results in results_dict.items():
            if 'actuals' in results and 'predictions' in results:
                metrics = calculate_metrics(results['actuals'], results['predictions'])
                score = metrics['r2']  # Use R² as main ranking metric
                model_rankings.append((model_name, score, metrics))
        
        model_rankings.sort(key=lambda x: x[1], reverse=True)
        
        html_content += """
                    <div class="model-comparison">
                        <h3><span class="success-icon">🏆</span>Model Ranking</h3>
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
    
    # Detailed Metrics Section
    html_content += """
                <div id="metrics" class="section">
                    <h2>📊 Detailed Metrics</h2>
                    
                    <div class="tabs">
                        <button class="tab active" onclick="showTab('metrics-overview')">Overview</button>
                        <button class="tab" onclick="showTab('detailed-metrics')">Detailed Analysis</button>
                    </div>
                    
                    <div id="metrics-overview" class="tab-content active">
    """
    
    # Metric cards for each model
    for model_name, results in results_dict.items():
        if 'actuals' in results and 'predictions' in results:
            metrics = calculate_metrics(results['actuals'], results['predictions'])
            
            html_content += f"""
                        <h3>{model_name}</h3>
                        <div class="metric-grid">
            """
            
            # R² Card
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
            
            # RMSE Card
            html_content += f"""
                            <div class="metric-card">
                                <div class="metric-title">
                                    {explanations['rmse']['icon']} {explanations['rmse']['name']}
                                    <button class="toggle-btn" onclick="toggleExplanation('rmse-explanation-{model_name.replace(" ", "_")}')">?</button>
                                </div>
                                <div class="metric-value" style="color: #3498db">
                                    {metrics['rmse']:.6f}
                                </div>
                                <div class="metric-explanation">Lower is better</div>
                                <div id="rmse-explanation-{model_name.replace(" ", "_")}" class="explanation-panel">
                                    {explanations['rmse']['explanation']}
                                </div>
                            </div>
            """
            
            # MAE Card
            html_content += f"""
                            <div class="metric-card">
                                <div class="metric-title">
                                    {explanations['mae']['icon']} {explanations['mae']['name']}
                                    <button class="toggle-btn" onclick="toggleExplanation('mae-explanation-{model_name.replace(" ", "_")}')">?</button>
                                </div>
                                <div class="metric-value" style="color: #9b59b6">
                                    {metrics['mae']:.6f}
                                </div>
                                <div class="metric-explanation">Lower is better</div>
                                <div id="mae-explanation-{model_name.replace(" ", "_")}" class="explanation-panel">
                                    {explanations['mae']['explanation']}
                                </div>
                            </div>
            """
            
            # MSE Card
            html_content += f"""
                            <div class="metric-card">
                                <div class="metric-title">
                                    {explanations['mse']['icon']} {explanations['mse']['name']}
                                    <button class="toggle-btn" onclick="toggleExplanation('mse-explanation-{model_name.replace(" ", "_")}')">?</button>
                                </div>
                                <div class="metric-value" style="color: #3498db">
                                    {metrics['mse']:.6f}
                                </div>
                                <div class="metric-explanation">Lower is better</div>
                                <div id="mse-explanation-{model_name.replace(" ", "_")}" class="explanation-panel">
                                    {explanations['mse']['explanation']}
                                </div>
                            </div>
            """
            
            # MAPE Card (if available)
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
            
            # NMSE1 Card
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
                                <div class="metric-explanation">Lower is better (normalized)</div>
                                <div id="eqmn1-explanation-{model_name.replace(" ", "_")}" class="explanation-panel">
                                    {explanations['eqmn1']['explanation']}
                                </div>
                            </div>
                """
            
            # NMSE2 Card
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
                                <div class="metric-explanation">Less than 1.0 is better (outperforms naive model)</div>
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
                    
                    <div id="detailed-metrics" class="tab-content">
                        <div class="highlight-box">
                            <h3><span class="info-icon">📚</span>Advanced Interpretation of Metrics</h3>
                            <p>This section provides a deeper analysis of the meaning of each metric in the context of your machine learning model.</p>
                        </div>
                        
                        <h4>🎯 How to Interpret the Results</h4>
                        <ul>
                            <li><strong>R² close to 1:</strong> Indicates that the model explains almost all of the data variability</li>
                            <li><strong>RMSE low:</strong> Small prediction errors relative to data scale</li>
                            <li><strong>MAE low:</strong> Small mean absolute error, more robust to outliers</li>
                            <li><strong>MAPE low:</strong> Small percentage error, useful for comparison between different scales</li>
                        </ul>
                    </div>
                </div>
    """
    
    # Seção de Visualizações
    html_content += """
                <div id="charts" class="section">
                    <h2>📈 Visualizations</h2>
                    
                    <div class="highlight-box">
                        <h3><span class="info-icon">📊</span>Explanation Graphics</h3>
                        <p>The graphics below show different aspects of model performance. Click on each image to view in full screen.</p>
                    </div>
                    
                    <div class="tabs">
                        <button class="tab active" onclick="showTab('comparison-graphics')">📊 Comparison Graphics</button>
                        <button class="tab" onclick="showTab('individual-analyses')">🔍 Individual Analyses</button>
                    </div>
                    
                    <div id="comparison-graphics" class="tab-content active">
                        <h3>🌐 Comparison Analyses</h3>
                        <div class="metric-grid">
    """
    
    # Adicionar gráficos comparativos
    comparative_charts = ['overview', 'metrics_comparison', 'metrics_table']
    for file_key, file_path in generated_files.items():
        if any(keyword in file_key for keyword in comparative_charts) and file_path.endswith('.png'):
            file_name = os.path.basename(file_path)
            
            title_map = {
                'overview': '🌐 General Comparison View',
                'metrics_comparison': '📊 Metrics Comparison',
                'metrics_table': '📋 Metrics Table'
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
                    
                    <div id="individual-analyses" class="tab-content">
                        <h3>🔍 Individual Analyses</h3>
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
                    'training': '📈 Training History',
                    'predictions': '🎯 Predictions vs Actual Values',
                    'qq': '📊 Q-Q Plot Analysis',
                    'cdf': '📋 Cumulative Distribution Function',
                    'pdf': '📊 Probability Density Function',
                    'ks': '🔬 Kolmogorov-Smirnov Test',
                    'residuals': '📉 Residuals Analysis',
                    'autocorr': '🔄 Autocorrelation'
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
                <div id="analyses" class="section">
                    <h2>📈 Statistical Analysis</h2>
                    
                    <div class="highlight-box">
                        <h3><span class="info-icon">🧪</span>Advanced Tests and Analyses</h3>
                        <p>This section presents in-depth statistical analyses of model results.</p>
                    </div>
                    
                    <h3>📊 Available Analyses</h3>
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-title">📊 Q-Q Plot</div>
                            <div class="metric-explanation">
                                Compares the residuals distribution with the normal distribution. 
                                Points close to the diagonal line indicate normal residuals.
                            </div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-title">📈 Distribution Analysis</div>
                            <div class="metric-explanation">
                                Compares the distribution functions (CDF) and densities (PDF) between 
                                actual values and predictions.
                            </div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-title">🔬 Kolmogorov-Smirnov Test</div>
                            <div class="metric-explanation">
                                Tests if the distributions of actual values and predictions are 
                                statistically similar.
                            </div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-title">🔄 Autocorrelation Analysis</div>
                            <div class="metric-explanation">
                                Examines temporal dependence in data, important for 
                                time series.
                            </div>
                        </div>
                    </div>
                </div>
    """
    
    # Seção Guia de Métricas
    html_content += """
                <div id="guide" class="section">
                    <h2>📚 Metrics Guide</h2>
                    
                    <div class="highlight-box">
                        <h3><span class="info-icon">🎓</span>Learn About Machine Learning Metrics</h3>
                        <p>This guide explains all metrics used in model evaluation in detail.</p>
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
                        <h3><span class="success-icon">💡</span>Interpretation Tips</h3>
                        <ul>
                            <li><strong>Use multiple metrics:</strong> Each metric offers a different perspective</li>
                            <li><strong>Consider the context:</strong> The importance of each metric depends on the application</li>
                            <li><strong>Analyze residuals:</strong> Patterns in errors can reveal problems in the model</li>
                            <li><strong>Compare with baselines:</strong> Evaluate if the model outperforms simple methods</li>
                            <li><strong>Validate on new data:</strong> Performance on test is crucial</li>
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
                console.log('Interactive HTML Report loaded!');
                
                // Adicionar contador de cliques em explicações
                let explanationClicks = 0;
                document.querySelectorAll('.toggle-btn').forEach(btn => {{
                    btn.addEventListener('click', function() {{
                        explanationClicks++;
                        console.log(`Explanation viewed ${{explanationClicks}} times`);
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
    
    print(f"✅ Interactive HTML Report generated: {save_path}")
    return save_path 